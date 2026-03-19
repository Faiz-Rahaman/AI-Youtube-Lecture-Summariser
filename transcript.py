"""
transcript.py — YouTube transcript extraction module.
Extracts auto-generated or manually-added captions from a YouTube video.
"""

import re
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(youtube_url: str) -> str:
    """
    Parse a YouTube video ID from various URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    - https://youtube.com/shorts/VIDEO_ID
    """
    patterns = [
        r'(?:v=|\/v\/|\/embed\/|youtu\.be\/|\/shorts\/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {youtube_url}")


def extract_transcript(youtube_url: str) -> dict:
    """
    Fetch the transcript for a YouTube video.

    Returns:
        dict with keys:
            - video_id (str): The YouTube video ID
            - transcript_text (str): Full transcript as a single string
            - segments (list[dict]): Raw segments with text, start, duration
    """
    video_id = extract_video_id(youtube_url)

    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)

    segments = []
    for snippet in transcript:
        segments.append({
            "text": snippet.text,
            "start": snippet.start,
            "duration": snippet.duration,
        })

    full_text = " ".join(seg["text"] for seg in segments)

    return {
        "video_id": video_id,
        "transcript_text": full_text,
        "segments": segments,
    }
