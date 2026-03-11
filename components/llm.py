from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()  # works locally

def get_llm(temperature: float = 0.9):
    # Try Streamlit Cloud secrets first, then fall back to .env
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    return ChatGroq(
        model="llama3-8b-8192",
        temperature=temperature,
        groq_api_key=api_key
    )