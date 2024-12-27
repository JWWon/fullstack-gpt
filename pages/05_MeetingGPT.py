import glob
import os
import openai
import streamlit as st
import subprocess
from pydub import AudioSegment
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser

st.set_page_config(page_title="MeetingGPT", page_icon="💬")

st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.
Get started by uploading a video file in the sidebar.
"""
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def extract_audio_from_video(_video_path: str):
    _audio_path = _video_path.replace(".mp4", ".mp3")
    if not os.path.exists(_video_path):
        subprocess.run(["ffmpeg", "-y", "-i", _video_path, "-vn", _audio_path])
    return _audio_path


ten_minutes = 10 * 60 * 1000


def slice_audio(_audio_path: str, chunk_duration: int = ten_minutes):
    track = AudioSegment.from_mp3(_audio_path)
    _chunk_dir = f"{os.path.dirname(_audio_path)}/chunks"
    name = os.path.basename(_audio_path).replace(".mp3", "")
    for i, chunk in enumerate(track[::chunk_duration]):
        chunk_path = f"{_chunk_dir}/{name}_{i}.mp3"
        if not os.path.exists(chunk_path):
            os.makedirs(os.path.dirname(chunk_path))
            chunk.export(chunk_path, format="mp3")
    return _chunk_dir


def transcribe_audio_chunks(_chunk_dir: str, transcript_path: str):
    if os.path.exists(transcript_path):
        return

    for chunk in sorted(glob.glob(f"{_chunk_dir}/*.mp3")):
        with open(chunk, "rb") as audio_chunk, open(transcript_path, "a") as text_file:
            st.write(chunk, text_file)
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", file=audio_chunk
            )
            text_file.write(transcript.text)


with st.sidebar:
    video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

base_path = "./.cache/meeting_files"
if video:
    with st.status("Loading video...") as status:
        video_path = f"{base_path}/{video.name}"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(video.read())

        status.update(label="Extracting audio...")
        audio_path = extract_audio_from_video(video_path)

        status.update(label="Slicing audio...")
        chunk_dir = slice_audio(audio_path)

        status.update(label="Transcribing audio...")
        transcript_path = f"{base_path}/{video.name.replace('.mp4', '.txt')}"
        transcribe_audio_chunks(chunk_dir, transcript_path)

    transcript_tab, summary_tab = st.tabs(["Transcript", "Summary"])

    with transcript_tab:
        with open(transcript_path, "r") as f:
            st.write(f.read())

    with summary_tab:
        submit = st.button("Summarize")
        if submit:
            loader = TextLoader(transcript_path)
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800, chunk_overlap=200
            )
            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY: 
                """
            )
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )

            refine_chain = prompt | llm | StrOutputParser()

            with st.status(label="Summarizing...") as status:

                def update_progress(i: int):
                    status.update(label=f"Summarizing... {i + 1}/{len(docs) - 1}")

                update_progress(0)
                summary = first_summary_chain.invoke({"text": docs[0].page_content})
                for i, doc in enumerate(docs[1:]):
                    update_progress(i)
                    summary = refine_chain.invoke(
                        {"existing_summary": summary, "context": doc.page_content}
                    )

            st.write(summary)
