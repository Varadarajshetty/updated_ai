import os
import re
import requests
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import yt_dlp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from flask_sqlalchemy import SQLAlchemy
from pydub import AudioSegment
import tempfile

# Load environment variables from .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, using system env vars

# Configure ffmpeg path for pydub (cross-platform)
if os.name == 'nt':  # Windows
    FFMPEG_PATH = os.path.expandvars(r"$LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin")
    if os.path.exists(FFMPEG_PATH):
        AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
        AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, "ffprobe.exe")
        os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")
# On Linux/Render, ffmpeg is installed via apt and available in PATH

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
AUDIO_FOLDER = "static/audio"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["AUDIO_FOLDER"] = AUDIO_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', f'sqlite:///{os.path.join(os.getcwd(), "users.db")}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = None

db = SQLAlchemy(app)

# API Keys from environment variables (REQUIRED for production)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set in environment variables!")
client = OpenAI(api_key=OPENAI_API_KEY)

# Groq API for ultra-fast Whisper transcription
from groq import Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

HF_MODEL = "facebook/bart-large-cnn"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_ASR_URL = "https://api.openai.com/v1/audio/transcriptions"

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(150), db.ForeignKey('user.email'), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # 'pdf' or 'youtube'
    title = db.Column(db.String(200), nullable=False)
    summary = db.Column(db.Text, nullable=False)
    audio_filename = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# Create database tables
with app.app_context():
    db.create_all()

@app.route('/starter')
def starter():
    if 'user' not in session:
        return redirect(url_for('index'))
    user_name = session.get("user")
    return render_template("index.html", user_name=user_name)


@app.route('/logout')
def logout():
    """Clear session and redirect to login page."""
    session.clear()
    return redirect(url_for('index'))


@app.route('/')
def landing():
    """Landing page for non-logged in users."""
    if 'user' in session:
        return redirect(url_for('starter'))
    return render_template("landing.html")

########################## After Login #################################
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_with_vision(pdf_path):
    """Extract text from image-based PDF using OpenAI Vision API (optimized for speed)."""
    import fitz  # PyMuPDF for PDF to image conversion
    import base64
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print(f"[OCR] Starting Vision extraction...")
    
    def process_page(page_num, page_data):
        """Process a single page - runs in parallel."""
        try:
            # Render page to image (150 DPI - faster)
            mat = fitz.Matrix(150/72, 150/72)
            pix = page_data.get_pixmap(matrix=mat)
            
            # Convert to JPEG (smaller file size)
            img_bytes = pix.tobytes("jpeg")
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            
            # Send to OpenAI Vision API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an OCR system. Extract and output all visible text from the document image. Output ONLY the text content."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )
            page_text = response.choices[0].message.content
            
            # Filter out refusal responses
            refusal_phrases = ["i'm sorry", "i cannot", "i can't", "unable to", "i apologize", 
                             "i'm unable", "cannot extract", "can't extract", "no text visible",
                             "provide me", "appears to be", "don't see any", "image doesn't contain"]
            
            if page_text and not any(phrase in page_text.lower() for phrase in refusal_phrases):
                print(f"[OCR] Page {page_num + 1}: OK")
                return (page_num, page_text)
            else:
                print(f"[OCR] Page {page_num + 1}: Refusal")
                return (page_num, "")
                
        except Exception as e:
            print(f"[OCR] Page {page_num + 1} error: {e}")
            return (page_num, "")
    
    try:
        doc = fitz.open(pdf_path)
        num_pages = min(len(doc), 5)  # Limit to 5 pages
        print(f"[OCR] Processing {num_pages} pages in parallel...")
        
        # Prepare page data (can't pass fitz page objects directly to threads)
        pages = [(i, doc[i]) for i in range(num_pages)]
        
        # Process pages in parallel (3 concurrent requests)
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(process_page, page_num, page): page_num 
                      for page_num, page in pages}
            for future in as_completed(futures):
                results.append(future.result())
        
        doc.close()
        
        # Sort by page number and combine
        results.sort(key=lambda x: x[0])
        all_text = [text for _, text in results if text]
        result_text = "\n\n".join(all_text)
        print(f"[OCR] Done: {len(result_text)} chars from {len(all_text)} pages")
        return result_text
        
    except Exception as e:
        print(f"[OCR] Error: {e}")
        return ""


def extract_text(pdf_path):
    """Extract text from PDF with pdfplumber, fallback to OpenAI Vision for scanned/image PDFs."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except:
        pass

    # If text extraction failed, try OpenAI Vision API
    if len(text.strip()) < 50:
        print("Text extraction failed, trying OpenAI Vision API...")
        vision_text = extract_text_with_vision(pdf_path)
        if vision_text and len(vision_text.strip()) > 10:
            text = vision_text
            print(f"Successfully extracted {len(text)} characters using Vision API")
        else:
            # Vision API also failed
            if len(text.strip()) < 10:
                text = "[Error] Could not extract text from this PDF. The document may be empty or corrupted."

    return text.strip()


def clean_text(raw_text):
    """Clean headers, footers, page numbers."""
    text = re.sub(r'\bPage \d+\b', '', raw_text)
    text = re.sub(r'Table \d+.*?\n', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def format_summary_with_styling(content):
    """Format summary with clean, professional styling including headers."""
    import re
    
    # Convert markdown headers to HTML
    content = re.sub(r'^## (.+)$', r'<h3 style="color: #667eea; margin-top: 20px; margin-bottom: 10px; font-weight: 600;">\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'^### (.+)$', r'<h4 style="color: #764ba2; margin-top: 15px; margin-bottom: 8px; font-weight: 500;">\1</h4>', content, flags=re.MULTILINE)
    
    # Convert markdown formatting to HTML
    content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
    
    lines = content.split("\n")
    formatted_output = []
    current_list = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if it's a header (already converted to HTML)
        if line.startswith('<h3') or line.startswith('<h4'):
            # Close any open list
            if current_list:
                formatted_output.append(f"<ul style='padding-left: 20px; margin-bottom: 15px;'>{''.join(current_list)}</ul>")
                current_list = []
            formatted_output.append(line)
            continue
        
        # Remove leading bullet characters
        if line.startswith("-") or line.startswith("*") or line.startswith("•"):
            line = line[1:].strip()
        
        # Skip very short lines
        if len(line) < 5:
            continue
        
        # Add as list item
        current_list.append(f"<li style='margin-bottom: 8px; line-height: 1.7;'>{line}</li>")
    
    # Close final list
    if current_list:
        formatted_output.append(f"<ul style='padding-left: 20px;'>{''.join(current_list)}</ul>")
    
    return f"<div style='font-size: 1rem;'>{''.join(formatted_output)}</div>"


def summarize_with_huggingface(text):
    """Summarize text with comprehensive, structured formatting and high fidelity."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are an expert content summarizer. Create a comprehensive, well-structured summary following this format:

## Summary of Content
[Write 2-3 sentences providing an overview - USE EXACT PHRASES from the original text]

## Core Concepts and Definitions
- **[Term 1]**: [Use the EXACT definition from the source text]
- **[Term 2]**: [Copy key phrases VERBATIM from the original]
[Add more as needed]

## Key Points and Details
- **[Main Topic 1]**: [Copy important sentences and phrases DIRECTLY from the source]
- **[Main Topic 2]**: [Include specific facts, numbers, examples using original wording]
[Continue for 6-10 detailed points]

## Key Insights
- [Copy key conclusions VERBATIM from the text]
- [Use exact terminology from the source]

CRITICAL FOR HIGH ACCURACY:
1. Copy key sentences and phrases VERBATIM from the original text
2. Preserve ALL original terminology, names, and technical terms
3. Do NOT paraphrase when the original wording is precise
4. Use **bold** for key terms copied from the source"""},
                {"role": "user", "content": f"Create a structured summary. COPY key phrases verbatim from this text:\n\n{text[:6000]}"}
            ],
            temperature=0.2,
            max_tokens=1400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[OpenAI Summarization Error] {str(e)}"


def rewrite_narration_with_openai(original_text, summary):
    """Create a comprehensive, structured summary with high fidelity to original."""
    prompt = f"""Create a comprehensive, well-structured summary following this format:

## Summary of Content
[Write 2-3 sentences - USE EXACT PHRASES from the original text]

## Core Concepts and Definitions
- **[Term 1]**: [Use the EXACT definition from the source]
- **[Term 2]**: [Copy key phrases VERBATIM]
[Add more as needed]

## Key Points and Details
- **[Main Topic 1]**: [Copy important sentences DIRECTLY from the source]
- **[Main Topic 2]**: [Include facts using original wording]
[Continue for 6-10 detailed points]

## Key Insights
- [Copy key conclusions VERBATIM]
- [Use exact terminology from source]

CRITICAL: Copy key sentences and phrases VERBATIM from the original. Preserve ALL terminology.

Content:
{original_text[:6000]}

Notes:
{summary}
"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1600,
        temperature=0.2
    )
    content = completion.choices[0].message.content
    return format_summary_with_styling(content)


def generate_narration_script(text):
    """Optimized: Single API call for fast PDF summarization."""
    if not text:
        return "<div class='error-message' style='color: #e53e3e; padding: 20px; background: #fed7d7; border-radius: 8px;'><strong>Error:</strong> No readable text found in PDF.</div>"
    
    # Check if text is an error message (from failed extraction)
    if text.startswith("[Error]") or "could not extract text" in text.lower():
        return "<div class='error-message' style='color: #e53e3e; padding: 20px; background: #fed7d7; border-radius: 8px;'><strong>Error:</strong> Unable to extract text from this PDF. It may contain scanned images that could not be processed. Please try uploading a text-based PDF or ensure the image quality is clear.</div>"

    clean_notes = clean_text(text)
    
    # SPEED OPTIMIZATION: Single API call instead of two
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Create a comprehensive, detailed summary with well-elaborated points:

## Summary of Content
[Write 3-4 sentences providing a thorough overview using EXACT phrases from the original text]

## Core Concepts and Definitions
- **[Term 1]**: [Provide a detailed explanation in 2-3 sentences using the EXACT definition from source]
- **[Term 2]**: [Elaborate with specific details and examples from the text]
[Add 5-8 detailed definitions]

## Key Points and Details
- **[Main Topic 1]**: [Elaborate in 3-4 sentences. Include specific facts, numbers, and examples.]
- **[Main Topic 2]**: [Detailed explanation in 3-4 sentences using original wording.]
[Continue for 8-12 well-elaborated points]

## Key Insights and Takeaways
- [Elaborate on this insight in 2-3 sentences]
- [Another detailed takeaway with specific conclusions]

IMPORTANT: Each point should be ELABORATED with 2-4 sentences. Use **bold** for all key terms."""},
                {"role": "user", "content": f"Create a detailed, well-elaborated summary:\n\n{clean_notes[:10000]}"}
            ],
            temperature=0.2,
            max_tokens=2000  # Increased for more elaborated content
        )
        return format_summary_with_styling(response.choices[0].message.content)
    except Exception as e:
        return f"[Error: {e}]"



def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None


def get_transcript(video_id):
    """Retrieve transcript from YouTube API (v1.2.3+ uses fetch/list API)."""
    try:
        # New API in youtube-transcript-api v1.2.3+
        # Use YouTubeTranscriptApi.fetch() for direct transcript fetching
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        text = " ".join([entry.text for entry in transcript])
        print(f"Got transcript via YouTube API ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"Transcript API error: {e}")
        raise e


def download_audio(youtube_url):
    output_file = "audio_temp"  # yt-dlp will add extension
    ydl_opts = {
        # Select best audio but limit to 64k for faster download of multi-hour videos
        "format": "bestaudio[abr<=64]/bestaudio/best",
        "outtmpl": output_file,
        "quiet": True,
        "no_warnings": False,
        "ignoreerrors": False,
        "retries": 5,
        "fragment_retries": 5,
        "http_chunk_size": 10485760,  # 10MB chunks
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "referer": "https://www.youtube.com/",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "32",  # Fast transcription quality
        }],
        # REMOVED: Specific extractor_args for ios/player_client that were causing 403s
        # We'll use default clients which are usually more stable unless specifically blocked
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return "audio_temp.mp3"


def split_audio_file(audio_file, chunk_length_ms=600000):
    """
    Split audio file into chunks of specified length (default 10 minutes = 600000 ms).
    Returns list of temporary file paths for each chunk.
    Raises exception if ffmpeg is not available.
    """
    try:
        audio = AudioSegment.from_file(audio_file)
        chunks = []
        chunk_files = []
        
        # Split into chunks
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            # Create temporary file for chunk
            chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix='.m4a')
            chunk.export(chunk_file.name, format="m4a")
            chunk_files.append(chunk_file.name)
            chunks.append(chunk)
        
        return chunk_files
    except FileNotFoundError as e:
        error_msg = str(e)
        if "ffmpeg" in error_msg.lower() or "ffprobe" in error_msg.lower() or "cannot find the file" in error_msg.lower():
            raise Exception("ffmpeg is not installed. Please install ffmpeg to split large audio files. "
                          "Download from https://ffmpeg.org/download.html or use: choco install ffmpeg")
        raise
    except Exception as e:
        print(f"Error splitting audio: {e}")
        raise


def audio_to_text(audio_file):
    """
    Convert audio to text using Groq Whisper API (10x faster).
    Falls back to OpenAI Whisper if Groq is not configured.
    """
    file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
    print(f"Transcribing audio file: {file_size_mb:.2f} MB")
    
    # TRY GROQ FIRST (10x faster than OpenAI)
    if groq_client:
        try:
            # Groq has a 25MB limit. If file is larger, use the same chunking logic as OpenAI
            if file_size_mb < 24:
                print("Using Groq Whisper (ultra-fast)...")
                with open(audio_file, "rb") as audio_file_obj:
                    transcript = groq_client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio_file_obj,
                        response_format="text"
                    )
                print("Groq transcription complete!")
                return transcript
            else:
                print(f"File too large for Groq ({file_size_mb:.2f} MB), will use chunking in OpenAI fallback or we could implement Groq chunking.")
                # Fall through to OpenAI which has robust chunking
        except Exception as e:
            print(f"Groq error: {e}, falling back to OpenAI...")
    
    # FALLBACK TO OPENAI
    print("Using OpenAI Whisper...")
    max_size_mb = 20
    
    if file_size_mb < max_size_mb:
        try:
            with open(audio_file, "rb") as audio_file_obj:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=(os.path.basename(audio_file), audio_file_obj, "audio/m4a")
                )
            return transcript.text
        except Exception as e:
            error_msg = str(e)
            if "too large" in error_msg.lower() or "entity too large" in error_msg.lower():
                print(f"File too large, splitting into chunks...")
            else:
                return f"[OpenAI ASR Error] {error_msg}"
    
    # If file is large or direct processing failed, split into chunks
    print(f"Processing large audio file ({file_size_mb:.2f} MB), splitting into chunks...")
    
    try:
        chunk_files = split_audio_file(audio_file, chunk_length_ms=600000)  # 10 minute chunks
    except Exception as e:
        # If splitting fails (e.g., ffmpeg not installed), try processing the whole file anyway
        error_msg = str(e)
        if "ffmpeg" in error_msg.lower():
            print(f"Warning: {error_msg}")
            print("Attempting to process the entire file directly (may fail if file is too large)...")
            try:
                with open(audio_file, "rb") as audio_file_obj:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=(os.path.basename(audio_file), audio_file_obj, "audio/m4a")
                    )
                return transcript.text
            except Exception as direct_error:
                return f"[OpenAI ASR Error] File is too large ({file_size_mb:.2f} MB) and cannot be split because ffmpeg is not installed. " \
                       f"Please install ffmpeg (https://ffmpeg.org/download.html) or use a shorter video. " \
                       f"Direct processing error: {str(direct_error)}"
        else:
            return f"[OpenAI ASR Error] Failed to split audio: {error_msg}"
    
    all_transcripts = []
    temp_files_to_cleanup = []
    
    try:
        for i, chunk_file in enumerate(chunk_files):
            print(f"Processing chunk {i+1}/{len(chunk_files)}...")
            try:
                with open(chunk_file, "rb") as chunk_file_obj:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=(os.path.basename(chunk_file), chunk_file_obj, "audio/m4a")
                    )
                chunk_text = transcript.text
                if chunk_text:
                    all_transcripts.append(chunk_text)
            except Exception as e:
                error_msg = str(e)
                print(f"Error processing chunk {i+1}: {error_msg}")
                # Continue with other chunks even if one fails
                continue
            finally:
                # Mark for cleanup
                if chunk_file != audio_file:  # Don't delete original file
                    temp_files_to_cleanup.append(chunk_file)
        
        # Clean up temporary chunk files
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error cleaning up {temp_file}: {e}")
        
        if not all_transcripts:
            return "[OpenAI ASR Error] Failed to process any audio chunks. The audio file may be too large or corrupted."
        
        # Combine all transcripts
        combined_text = " ".join(all_transcripts)
        print(f"Successfully processed {len(all_transcripts)} chunks")
        return combined_text
        
    except Exception as e:
        # Clean up on error
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        return f"[OpenAI ASR Error] Failed to process audio: {str(e)}"


def summarize_text(text):
    """SPEED OPTIMIZED: Fast structured summarization with comprehensive output."""
    CHUNK_SIZE = 12000
    MAX_INPUT = 150000  # Increased for 2+ hour videos
    
    text = text[:MAX_INPUT]
    
    # Comprehensive system prompt with MORE ELABORATION
    SYSTEM_PROMPT = """Create a comprehensive, detailed summary with well-elaborated points:

## Summary of Content
[Write 3-4 sentences providing a thorough overview using EXACT phrases from the original text]

## Core Concepts and Definitions
- **[Term 1]**: [Provide a detailed explanation in 2-3 sentences using the EXACT definition from source]
- **[Term 2]**: [Elaborate with specific details and examples from the text]
- **[Term 3]**: [Another key concept with full explanation]
[Add 5-8 detailed definitions]

## Key Points and Details
- **[Main Topic 1]**: [Elaborate in 3-4 sentences. Copy important sentences DIRECTLY from the source. Include specific facts, numbers, and examples.]
- **[Main Topic 2]**: [Detailed explanation in 3-4 sentences using original wording. Include all relevant details.]
- **[Main Topic 3]**: [Another elaborated point with specific information]
[Continue for 8-12 well-elaborated points]

## Key Insights and Takeaways
- [Elaborate on this insight in 2-3 sentences using exact terminology]
- [Another detailed takeaway with specific conclusions]
- [Third insight with supporting details]

IMPORTANT: Each point should be ELABORATED with 2-4 sentences. Copy key phrases VERBATIM. Use **bold** for all key terms."""

    try:
        # SPEED: Single API call for most content
        if len(text) <= CHUNK_SIZE:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Create a detailed, well-elaborated summary:\n\n{text}"}
                ],
                temperature=0.2,
                max_tokens=2000  # Increased for more elaborated content
            )
            content = response.choices[0].message.content
        else:
            # SPEED: Max 12 chunks for 2-hour videos (approx 150k chars)
            chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)][:12]
            chunk_summaries = []
            
            for chunk in chunks:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Extract key concepts, definitions, and detailed points. COPY phrases verbatim. Use **bold**."},
                        {"role": "user", "content": f"Extract detailed key points:\n\n{chunk}"}
                    ],
                    temperature=0.2,
                    max_tokens=800  # Restored for more content
                )
                chunk_summaries.append(response.choices[0].message.content)
            
            # Single final API call
            combined = "\n\n".join(chunk_summaries)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Combine into comprehensive summary:\n\n{combined}"}
                ],
                temperature=0.2,
                max_tokens=1500  # Restored for comprehensive output
            )
            content = response.choices[0].message.content
            
    except Exception as e:
        return f"<p class='text-red-600'>[Error] {str(e)}</p>"

    return format_summary_with_styling(content)


def generate_tts_audio(text, filename):
    """Generate TTS audio file from text using pyttsx3. Skips for very long text."""
    if len(text) > 10000:
        print("Summary too long for TTS, skipping to save time.")
        return None
    try:
        import pythoncom
        pythoncom.CoInitialize()
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        filepath = os.path.join(app.config["AUDIO_FOLDER"], filename)
        os.makedirs(app.config["AUDIO_FOLDER"], exist_ok=True)
        engine.save_to_file(text, filepath)
        engine.runAndWait()
        pythoncom.CoUninitialize()
        return filename
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


def generate_flashcards(summary):
    """Generate MCQs and one-sentence questions from summary using OpenAI."""
    prompt = f"""
    Based on the following summary, generate 5 multiple-choice questions (MCQs) and 5 one-sentence questions. Each MCQ should have 4 options (A, B, C, D) with one correct answer. Each one-sentence question should be a short answer question.

    Summary:
    {summary[:3000]}  # Limit to avoid token issues

    Output only the JSON in the following format, no additional text:
    {{
        "mcqs": [
            {{
                "question": "Question text?",
                "options": ["A) Option1", "B) Option2", "C) Option3", "D) Option4"],
                "correct": "A"
            }},
            ...
        ],
        "short_questions": [
            {{
                "question": "One-sentence question?",
                "answer": "Expected short answer"
            }},
            ...
        ]
    }}
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        content = completion.choices[0].message.content.strip()
        # Clean the content to extract JSON
        import json
        import re
        # Find JSON-like content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        flashcards = json.loads(content)
        return flashcards
    except Exception as e:
        return {"error": str(e)}


def evaluate_summary_metrics(original_text, summary_text):
    from math import isnan

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(original_text, summary_text)

    rouge1 = rouge_scores['rouge1'].fmeasure or 0
    rouge2 = rouge_scores['rouge2'].fmeasure or 0
    rougeL = rouge_scores['rougeL'].fmeasure or 0

    vectorizer = TfidfVectorizer(stop_words='english').fit([original_text, summary_text])
    vectors = vectorizer.transform([original_text, summary_text])
    cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Clean invalid values
    for v in [rouge1, rouge2, rougeL, cosine_sim]:
        if isnan(v):
            v = 0

    overall = (rouge1 + rouge2 + rougeL + cosine_sim) / 4 * 10

    return {
        "rouge1": round(rouge1, 3),
        "rouge2": round(rouge2, 3),
        "rougeL": round(rougeL, 3),
        "cosine": round(cosine_sim, 3),
        "overall": round(overall, 2)
    }



@app.route("/login", methods=["GET", "POST"])
def index():
    return render_template("login.html")


@app.route("/login_validation", methods=["POST"])
def login_validation():
    email = request.form["email"]
    password = request.form["password"]
    user = User.query.filter_by(email=email).first()
    if user and user.password == password:
        session["user"] = email
        flash("Login successful!", "success")
        return redirect(url_for("starter"))
    else:
        flash("Invalid email or password.", "danger")
        return redirect(url_for("index"))


@app.route("/chat", methods=["POST"])
def chat():
    """AI Assistant chat endpoint - answers questions about the document."""
    from flask import jsonify
    
    data = request.get_json()
    user_message = data.get("message", "")
    context = data.get("context", "")
    
    if not user_message:
        return jsonify({"response": "Please enter a message."})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a helpful AI assistant. The user has just read a document summary. 
Answer their questions based on the document content below. Be concise and helpful.
If the question is not related to the document, still try to help but mention that it's outside the document scope.

Document Summary:
{context[:8000]}"""
                },
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        assistant_response = response.choices[0].message.content
        return jsonify({"response": assistant_response})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"response": "Sorry, I encountered an error. Please try again."})


@app.route("/pdf", methods=["POST"])
def pdf_upload():
    if "file" not in request.files:
        return redirect("/")
    file = request.files["file"]
    if file.filename == "":
        return redirect("/")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(filepath)

        text = extract_text(filepath)
        narration_script = generate_narration_script(text)

        metrics = evaluate_summary_metrics(text, narration_script)
        audio_filename = generate_tts_audio(narration_script, f"summary_{filename}.mp3")

        # Save to history
        user_email = session.get("user")
        if user_email:
            history_entry = History(
                user_email=user_email,
                type="pdf",
                title=filename,
                summary=narration_script,
                audio_filename=audio_filename
            )
            db.session.add(history_entry)
            db.session.commit()

        flashcards = generate_flashcards(narration_script)
        return render_template("result.html", summary=narration_script, metrics=metrics, audio=audio_filename, flashcards=flashcards)

    return redirect("/")


# Allowed video extensions
VIDEO_EXTENSIONS = {"mp4", "mkv", "avi", "mov", "webm", "m4v"}

def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in VIDEO_EXTENSIONS


def extract_audio_from_video(video_path):
    """Extract FULL audio with maximum compression for fast upload to Whisper."""
    import subprocess
    
    audio_output = video_path.rsplit(".", 1)[0] + "_audio.mp3"
    
    try:
        ffmpeg_cmd = os.path.join(FFMPEG_PATH, "ffmpeg.exe") if os.path.exists(FFMPEG_PATH) else "ffmpeg"
        
        cmd = [
            ffmpeg_cmd,
            "-i", video_path,
            # NO -t flag = extract FULL audio
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-b:a", "32k",  # Very low bitrate for fast upload
            "-ar", "16000",  # 16kHz optimized for speech
            "-ac", "1",  # Mono
            "-y",
            audio_output
        ]
        
        # Timeout set to 7200 seconds (2 hours) for very long videos
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None
            
        return audio_output
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None


@app.route("/video", methods=["POST"])
def video_upload():
    """Handle video file uploads for summarization."""
    if "video" not in request.files:
        flash("No video file provided.", "danger")
        return redirect(url_for("starter"))
    
    file = request.files["video"]
    if file.filename == "":
        flash("No video file selected.", "danger")
        return redirect(url_for("starter"))
    
    if not allowed_video(file.filename):
        flash("Invalid video format. Please upload MP4, MKV, AVI, MOV, or WebM files.", "danger")
        return redirect(url_for("starter"))
    
    # Save uploaded video
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(video_path)
    
    print(f"Video uploaded: {filename}")
    
    try:
        # Extract audio from video
        print("Extracting audio from video...")
        audio_path = extract_audio_from_video(video_path)
        
        if not audio_path or not os.path.exists(audio_path):
            flash("Failed to extract audio from video. Make sure ffmpeg is installed.", "danger")
            if os.path.exists(video_path):
                os.remove(video_path)
            return redirect(url_for("starter"))
        
        # Transcribe audio
        print("Transcribing audio...")
        transcript = audio_to_text(audio_path)
        
        # Cleanup files
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(video_path):
            os.remove(video_path)
        
        # Check for errors
        if transcript and transcript.startswith("[OpenAI ASR Error]"):
            flash(f"Error transcribing video: {transcript}", "danger")
            return redirect(url_for("starter"))
        
        if not transcript or len(transcript.strip()) < 10:
            flash("Could not extract any speech from the video.", "warning")
            return redirect(url_for("starter"))
        
        # Generate summary
        print(f"Generating summary from {len(transcript)} chars of transcript...")
        summary = summarize_text(transcript)
        metrics = evaluate_summary_metrics(transcript, summary)
        # SPEED: Skip TTS for video uploads (saves ~5-10 seconds)
        audio_filename = None
        
        # Save to history
        user_email = session.get("user")
        if user_email:
            history_entry = History(
                user_email=user_email,
                type="video",
                title=filename,
                summary=summary,
                audio_filename=audio_filename
            )
            db.session.add(history_entry)
            db.session.commit()
        
        flashcards = generate_flashcards(summary)
        return render_template("result.html", summary=summary, metrics=metrics, audio=audio_filename, flashcards=flashcards)
        
    except Exception as e:
        print(f"Video processing error: {e}")
        # Cleanup on error
        if os.path.exists(video_path):
            os.remove(video_path)
        flash(f"Error processing video: {str(e)}", "danger")
        return redirect(url_for("starter"))


@app.route("/youtube", methods=["POST"])
def youtube_summary():
    youtube_url = request.form["youtube_url"]
    video_id = extract_video_id(youtube_url)
    
    if not video_id:
        flash("Invalid YouTube URL. Please check and try again.", "danger")
        return redirect(url_for("starter"))

    transcript = None
    error_msg = None
    
    # Try YouTube transcript first (fastest method)
    try:
        transcript = get_transcript(video_id)
        print(f"Got transcript via YouTube API ({len(transcript)} chars)")
    except Exception as e:
        print(f"YouTube transcript failed: {e}")
        error_msg = str(e)
    
    # Fallback to audio transcription only if transcript failed
    if not transcript:
        try:
            print("Falling back to audio download...")
            audio_file = download_audio(youtube_url)
            
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            print(f"Audio file size: {file_size_mb:.2f} MB")
            
            transcript = audio_to_text(audio_file)
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except Exception as e:
            print(f"Audio fallback failed: {e}")
            flash(f"Could not process video: {error_msg or str(e)}", "danger")
            return redirect(url_for("starter"))
    
    # Check if transcript is an error message
    if transcript and transcript.startswith("[OpenAI ASR Error]"):
        error_message = transcript
        flash(f"Error processing audio: {error_message}", "danger")
        return render_template("result.html", 
                             summary=f"<p class='text-red-600 font-semibold'>{error_message}</p><p>Please try a shorter video or ensure the audio file is not too large.</p>", 
                             metrics=None, 
                             audio=None, 
                             flashcards=None)
    
    if not transcript or len(transcript.strip()) < 10:
        flash("Failed to extract transcript from the video. Please try again or use a different video.", "danger")
        return redirect(url_for("starter"))

    summary = summarize_text(transcript)
    metrics = evaluate_summary_metrics(transcript, summary)
    audio_filename = generate_tts_audio(summary, f"summary_{video_id}.mp3")

    # Save to history
    user_email = session.get("user")
    if user_email:
        history_entry = History(
            user_email=user_email,
            type="youtube",
            title=f"YouTube Video {video_id}",
            summary=summary,
            audio_filename=audio_filename
        )
        db.session.add(history_entry)
        db.session.commit()

    flashcards = generate_flashcards(summary)
    return render_template("result.html", summary=summary, metrics=metrics, audio=audio_filename, flashcards=flashcards)


@app.route("/add_users", methods=["GET", "POST"])
def add_users():
    if request.method == "POST":
        email = request.form["uemail"]
        password = request.form["upassword"]
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already exists.")
            return redirect(url_for("add_users"))
        new_user = User(email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please log in.")
        return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/history")
def history():
    user_email = session.get("user")
    if not user_email:
        return {"history": []}
    history_entries = History.query.filter_by(user_email=user_email).order_by(History.created_at.desc()).all()
    history_data = [
        {
            "id": entry.id,
            "type": entry.type,
            "title": entry.title,
            "summary": entry.summary[:100] + "..." if len(entry.summary) > 100 else entry.summary,
            "audio_filename": entry.audio_filename,
            "created_at": entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for entry in history_entries
    ]
    return {"history": history_data}

@app.route("/history/<int:history_id>")
def view_history(history_id):
    user_email = session.get("user")
    if not user_email:
        return redirect(url_for("index"))
    history_entry = History.query.filter_by(id=history_id, user_email=user_email).first()
    if not history_entry:
        return "History item not found", 404
    return render_template("result.html", summary=history_entry.summary, audio=history_entry.audio_filename)


@app.route("/history/<int:history_id>/delete", methods=["POST"])
def delete_history(history_id):
    """Delete a single history item."""
    user_email = session.get("user")
    if not user_email:
        return {"success": False, "error": "Not logged in"}, 401
    
    history_entry = History.query.filter_by(id=history_id, user_email=user_email).first()
    if history_entry:
        db.session.delete(history_entry)
        db.session.commit()
        return {"success": True}
    return {"success": False, "error": "Not found"}, 404


@app.route("/history/clear-all", methods=["POST"])
def clear_all_history():
    """Delete all history for the current user."""
    user_email = session.get("user")
    if not user_email:
        return {"success": False, "error": "Not logged in"}, 401
    
    History.query.filter_by(user_email=user_email).delete()
    db.session.commit()
    return {"success": True}


@app.route("/generate_flashcards", methods=["POST"])
def generate_flashcards_route():
    summary = request.form.get("summary")
    if not summary:
        return {"error": "No summary provided"}
    flashcards = generate_flashcards(summary)
    return {"flashcards": flashcards}


@app.route("/submit_flashcards", methods=["POST"])
def submit_flashcards():
    data = request.get_json()
    mcq_answers = data.get("mcq_answers", {})
    short_answers = data.get("short_answers", {})

    score = 0
    total = 0

    # Assuming flashcards are passed or stored, but for simplicity, regenerate or assume
    # In real app, store flashcards in session or DB
    summary = data.get("summary", "")
    flashcards = generate_flashcards(summary)

    if "error" not in flashcards:
        # Score MCQs
        for i, mcq in enumerate(flashcards.get("mcqs", [])):
            total += 1
            user_answer = mcq_answers.get(str(i))
            if user_answer == mcq["correct"]:
                score += 1

        # Score short questions (simple string match, can be improved)
        for i, sq in enumerate(flashcards.get("short_questions", [])):
            total += 1
            user_answer = short_answers.get(str(i), "").strip().lower()
            correct_answer = sq["answer"].strip().lower()
            if user_answer == correct_answer:
                score += 1

    return {"score": score, "total": total}


if __name__ == "__main__":
    app.run(debug=True)
