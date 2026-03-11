from langchain_groq import ChatGroq
import streamlit as st
import os

def get_llm(temperature: float = 0.9):
    # Try Streamlit Cloud secrets first, then fall back to env variable
    api_key = None
    
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in secrets or environment!")

    return ChatGroq(
        model="llama3-8b-8192",
        temperature=temperature,
        groq_api_key=api_key
    )