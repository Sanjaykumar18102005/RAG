import os
import asyncio
import streamlit as st
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from config import GEMINI_MODEL, EMBEDDING_MODEL, VECTORSTORE_PATH

# ================================
# Utility Functions
# ================================
def build_vectorstore(pdf_file, save_path=VECTORSTORE_PATH):
    """Load PDF, split into chunks, create embeddings, and save FAISS vectorstore"""
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    st.info(f"✅ Loaded {len(documents)} pages from PDF")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    st.info(f"✂️ Split into {len(docs)} chunks")

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # Build FAISS vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save vectorstore
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vectorstore.save_local(save_path)
    st.success(f"💾 Vector store saved at: {save_path}")
    return vectorstore


def load_vectorstore(path=VECTORSTORE_PATH):
    """Load existing FAISS vector store with embeddings"""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # ✅ Required for pickle
    )
    return vectorstore


def build_qa_pipeline(vectorstore):
    """Build RetrievalQA pipeline using Gemini LLM and FAISS retriever"""
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return qa


# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="📄 Gemini RAG Q&A", page_icon="🤖", layout="wide")
st.title("📄 RAG-based PDF Q&A with Gemini")

uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    pdf_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF uploaded successfully!")

    if st.button("🔎 Process Document"):
        with st.spinner("Building vectorstore..."):
            vectorstore = build_vectorstore(pdf_path)
        st.success("✅ Document ready for Q&A!")

        # Save state
        st.session_state["vectorstore_ready"] = True

# Q&A Section
if "vectorstore_ready" in st.session_state and st.session_state["vectorstore_ready"]:
    vectorstore = load_vectorstore()
    qa = build_qa_pipeline(vectorstore)

    st.subheader("💬 Ask a Question from the Document")
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Generating answer..."):
            try:
                response = qa.invoke({"query": query})["result"]
                st.markdown(f"### 🤖 Answer:\n{response}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
