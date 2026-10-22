
import os
import hashlib
from pathlib import Path
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -------------------- CONFIGURATION --------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)
API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="DocuMind | Document AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- PROFESSIONAL UI --------------------

st.markdown("""
<style>
    .stApp {
        background: #F4F7FB;
        color: #172B4D;
    }

    .block-container {
        max-width: 1600px;
        padding: 1.5rem 2rem 2rem;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    .brand {
        font-size: 1.65rem;
        font-weight: 800;
        color: #15304B;
        letter-spacing: -0.5px;
    }

    .hero {
        background: linear-gradient(120deg, #15304B, #167D83);
        padding: 24px 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white;
        font-size: 1.8rem;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #E1F5F4;
        margin: 0;
        font-size: 0.95rem;
    }

    .panel-title {
        font-size: 1.08rem;
        font-weight: 750;
        color: #15304B;
        margin-bottom: 8px;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- EMBEDDINGS --------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -------------------- PDF PROCESSING --------------------

def process_pdf(pdf_bytes, filename):
    reader = PdfReader(BytesIO(pdf_bytes))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    texts = []
    metadata = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            page_chunks = splitter.split_text(page_text)

            for chunk in page_chunks:
                texts.append(chunk)
                metadata.append({"page": page_number})

    if not texts:
        raise ValueError(
            "No readable text found. The PDF may be scanned or image-based."
        )

    doc_hash = hashlib.md5(pdf_bytes).hexdigest()[:12]
    collection_name = f"doc_{doc_hash}"

    db = Chroma.from_texts(
        texts=texts,
        embedding=load_embeddings(),
        metadatas=metadata,
        collection_name=collection_name,
    )

    return db, len(texts), len(reader.pages)


# -------------------- QUESTION ANSWERING --------------------

def generate_answer(db, question):
    results = db.similarity_search(question, k=4)

    if not results:
        return (
            "The document does not provide enough information "
            "to answer that question."
        ), []

    context = "\n\n".join(
        f"PDF page {item.metadata.get('page', '?')}:\n"
        f"{item.page_content}"
        for item in results
    )

    client = Groq(api_key=API_KEY)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Use ONLY the supplied PDF excerpts. "
                    "If the excerpts do not contain enough information, "
                    "clearly say the document does not provide enough "
                    "information. Never invent facts. Include supporting "
                    "PDF page numbers in your answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PDF EXCERPTS:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "Answer clearly and cite the source page numbers."
                ),
            },
        ],
        temperature=0.2,
        max_completion_tokens=700,
    )

    answer = response.choices[0].message.content
    return answer, results


# -------------------- SESSION STATE --------------------

defaults = {
    "db": None,
    "pdf_bytes": None,
    "doc_info": None,
    "chat_history": [],
    "processed_hash": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">📚 DocuMind</div>',
        unsafe_allow_html=True,
    )
    st.caption("AI-powered document assistant")
    st.divider()

    st.markdown("### Your document")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Choose a text-based PDF to ask questions about.",
    )

    if uploaded_file:
        st.caption(f"Selected: {uploaded_file.name}")

        if st.button(
            "⚡ Process document",
            type="primary",
            use_container_width=True,
        ):
            pdf_bytes = uploaded_file.getvalue()
            pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

            with st.spinner("Reading and indexing your document..."):
                try:
                    db, chunk_count, page_count = process_pdf(
                        pdf_bytes,
                        uploaded_file.name,
                    )

                    st.session_state.db = db
                    st.session_state.pdf_bytes = pdf_bytes
                    st.session_state.doc_info = {
                        "name": uploaded_file.name,
                        "pages": page_count,
                        "chunks": chunk_count,
                    }
                    st.session_state.processed_hash = pdf_hash
                    st.session_state.chat_history = []

                    st.success("Document is ready!")

                except Exception as error:
                    st.error(f"PDF processing failed: {error}")

    st.divider()
    st.markdown("### Processing status")

    if st.session_state.doc_info:
        info = st.session_state.doc_info

        st.success("Document processed")
        st.caption(info["name"])

        metric1, metric2 = st.columns(2)
        metric1.metric("Pages", info["pages"])
        metric2.metric("Chunks", info["chunks"])

        if st.button("🗑️ Clear document", use_container_width=True):
            st.session_state.db = None
            st.session_state.pdf_bytes = None
            st.session_state.doc_info = None
            st.session_state.processed_hash = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Upload and process a PDF to begin.")

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq")


# -------------------- MAIN HEADER --------------------

st.markdown("""
<div class="hero">
    <h1>Your documents. Your answers.</h1>
    <p>
        Ask questions, explore key information, and verify answers
        against your uploaded PDF.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------- DASHBOARD --------------------

chat_column, document_column = st.columns(
    [1.1, 0.9],
    gap="large",
)


# -------------------- CHAT PANEL --------------------

with chat_column:
    st.markdown(
        '<div class="panel-title">💬 Document chat</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.caption(
            f"Ready to answer questions about {info['name']}"
        )
    else:
        st.info(
            "Upload and process a PDF from the sidebar to start chatting."
        )

    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.markdown(item["answer"])

            with st.expander("📎 View source passages"):
                if item["sources"]:
                    for source in item["sources"]:
                        page = source.metadata.get("page", "?")
                        st.markdown(f"**PDF page {page}**")
                        st.write(source.page_content)
                        st.divider()
                else:
                    st.caption("No source passages were retrieved.")

    question = st.chat_input(
        "Ask a question about your document...",
        disabled=(st.session_state.db is None),
    )

    if question:
        if not API_KEY:
            st.error(
                "GROQ_API_KEY is missing. Configure your API key "
                "in your local .env file or Streamlit Cloud secrets."
            )
        else:
            with st.spinner(
                "Searching the PDF and generating an answer..."
            ):
                try:
                    answer, sources = generate_answer(
                        st.session_state.db,
                        question,
                    )

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                    })

                    st.rerun()

                except Exception as error:
                    st.error(f"Could not generate an answer: {error}")

    if st.session_state.chat_history:
        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()


# -------------------- PDF DOCUMENT PREVIEW --------------------

with document_column:
    st.markdown(
        '<div class="panel-title">📄 Document preview</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.pdf_bytes:
        info = st.session_state.doc_info

        st.caption(
            f"{info['name']} · {info['pages']} pages"
        )

        # Native Streamlit PDF viewer instead of a Base64 iframe
        try:
            st.pdf(
                BytesIO(st.session_state.pdf_bytes),
                height=720,
            )
        except (AttributeError, TypeError):
            st.warning(
                "The PDF viewer is unavailable in this Streamlit version. "
                "You can download the PDF below."
            )

        st.download_button(
            "⬇️ Download original PDF",
            data=st.session_state.pdf_bytes,
            file_name=info["name"],
            mime="application/pdf",
            use_container_width=True,
        )

    else:
        st.markdown("""
        <div style="
            height:360px;
            background:white;
            border:1px dashed #CBD5E1;
            border-radius:14px;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
            padding:24px;
            color:#64748B;">
            <div>
                <div style="font-size:42px;">📄</div>
                <h3 style="color:#15304B;">Your PDF will appear here</h3>
                <p>Upload and process a document using the sidebar.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# -------------------- FOOTER --------------------

st.divider()
st.caption(
    "DocuMind · Retrieval-Augmented Generation (RAG) · "
    "Answers are grounded in retrieved document passages."
)
