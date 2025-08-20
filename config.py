import os
import streamlit as st

# ================================
# Google API Key (from Streamlit secrets)
# ================================
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found! Add it in Streamlit Secrets.")

# ================================
# Model choices
# ================================
GEMINI_MODEL = "gemini-1.5-flash"         # fast + cheap
EMBEDDING_MODEL = "models/text-embedding-004"  # for document embeddings

# ================================
# FAISS Vectorstore path
# ================================
VECTORSTORE_PATH = r"C:\python\RAG\vectorstore"
