import os
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

def test_transcript(video_id):
    print(f"Testing transcript for {video_id}...")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        print("Success! Transcript length:", len(transcript))
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

def test_ytdlp(url):
    print(f"Testing yt-dlp for {url}...")
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            print("Success! Title:", info.get('title'))
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

if __name__ == "__main__":
    video_id = "jNQXAC9IVRw" # Me at the zoo (short video)
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    test_transcript(video_id)
    test_ytdlp(url)
