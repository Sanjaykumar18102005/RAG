import os

# Set your Gemini API Key (from AI Studio or Vertex AI)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBzALzAl7wbe8pgX6F-tui8LPy1sBpyrCg"

# Model choices
GEMINI_MODEL = "gemini-1.5-flash"         # fast + cheap
EMBEDDING_MODEL = "models/text-embedding-004"  # for document embeddings
VECTORSTORE_PATH = r"C:\python\RAG\vectorstore"