import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import openai
import os

# --- Setup ---
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or replace with your key directly

# --- Helper Functions ---
def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript

def summarize_with_timestamps(transcript):
    summarized = []
    for i, chunk in enumerate(transcript[::10]):
        text = " ".join([t['text'] for t in transcript[i*10:i*10+10]])
        timestamp = chunk['start']
        summarized.append((timestamp, text))
    return summarized

def gpt_summarize(text):
    prompt = f"Summarize this lecture text in simple bullet points:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response['choices'][0]['message']['content']

def summarize_with_gpt(transcript):
    full_text = " ".join([t['text'] for t in transcript])
    return gpt_summarize(full_text)

def find_doubtful_sections(summarized_chunks):
    doubt_keywords = ['assume', 'therefore', 'hence', 'conclude', 'implies']
    doubts = []
    for ts, chunk in summarized_chunks:
        if any(word in chunk.lower() for word in doubt_keywords) or len(chunk.split()) > 30:
            doubts.append((ts, chunk))
    return doubts

def generate_mind_map(summarized_chunks):
    G = nx.Graph()
    keyword_map = defaultdict(list)

    for ts, chunk in summarized_chunks:
        words = chunk.lower().split()
        for word in words:
            if word.isalpha() and len(word) > 5:
                keyword_map[word].append(ts)

    top_keywords = sorted(keyword_map.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    for keyword, times in top_keywords:
        G.add_node(keyword)
        for t in times:
            G.add_node(str(round(t, 2)))
            G.add_edge(keyword, str(round(t, 2)))

    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=1000, font_size=10, ax=ax)
    plt.title("Conceptual Mind Map")
    return fig

# --- Streamlit App ---
st.title("AI YouTube Lecture Summariser")
video_id = st.text_input("Enter YouTube Video ID (e.g., gLZgdE2QvCo):")

if video_id:
    try:
        transcript = get_transcript(video_id)

        st.subheader("GPT Summary (Smarter 🤖)")
        gpt_summary = summarize_with_gpt(transcript)
        st.markdown(gpt_summary)

        summary = summarize_with_timestamps(transcript)
        doubts = find_doubtful_sections(summary)

        st.subheader("Interactive Summary Timeline")
        for ts, text in summary:
            st.markdown(f"**[{round(ts,2)} sec]** - {text}")

        st.subheader("Potential Doubt Sections")
        for ts, text in doubts:
            st.warning(f"[{round(ts,2)} sec] - {text}")

        st.subheader("Conceptual Mind Map")
        fig = generate_mind_map(summary)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Deployment Notes ---
# 1. Push this file and a requirements.txt to GitHub.
# 2. In requirements.txt include:
#    streamlit
youtube-transcript-api
networkx
matplotlib
openai
# 3. Deploy via https://streamlit.io/cloud linked to your repo.
