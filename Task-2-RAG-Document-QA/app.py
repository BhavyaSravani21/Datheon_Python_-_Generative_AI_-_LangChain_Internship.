
import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="DocuMind | Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# API KEY
# --------------------------------------------------

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("GROQ_API_KEY", "")


# --------------------------------------------------
# PROFESSIONAL UI STYLES
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Main page */
    .stApp {
        background: #f5f7fb;
        color: #172033;
    }

    [data-testid="stHeader"] {
        background: #f5f7fb;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e1e7f0;
    }

    [data-testid="stSidebar"] * {
        color: #172033;
    }

    /* Brand */
    .brand {
        font-size: 1.6rem;
        font-weight: 800;
        color: #304bb5 !important;
        margin-bottom: 0.2rem;
    }

    /* Hero banner */
    .hero {
        padding: 1.8rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #e5edff, #e3f7f4);
        border: 1px solid #d8e5f3;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #172554 !important;
        font-size: 2.2rem;
        margin: 0 0 0.5rem 0;
        font-weight: 800;
    }

    .hero p {
        color: #475569 !important;
        font-size: 1rem;
        margin: 0;
    }

    .section-title {
        color: #172554 !important;
        font-size: 1.2rem;
        font-weight: 750;
        margin-bottom: 0.8rem;
    }

    /* Information cards */
    .info-card {
        background: #ffffff;
        border: 1px solid #dbe4f0;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .info-card h3 {
        color: #172554 !important;
        margin-top: 0;
        font-size: 1.05rem;
    }

    .info-card p {
        color: #475569 !important;
        margin-bottom: 0;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e1e7f0;
        border-radius: 14px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        color: #172033 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 14px !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #172033 !important;
        caret-color: #304bb5 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }

    /* Standard input fields */
    .stTextInput input,
    .stTextArea textarea {
        background: #ffffff !important;
        color: #172033 !important;
    }

    /* ALL STANDARD BUTTONS */
    .stButton > button,
    .stDownloadButton > button {
        background-color: #e8efff !important;
        color: #243b80 !important;
        border: 1px solid #b8c9f0 !important;
        border-radius: 10px !important;
        font-weight: 650 !important;
        min-height: 2.6rem;
    }

    /* Button hover */
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #d5e2ff !important;
        color: #172554 !important;
        border-color: #829fe5 !important;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #304bb5 !important;
        color: #ffffff !important;
        border: 1px solid #304bb5 !important;
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #243b98 !important;
        color: #ffffff !important;
    }

    /* Disabled buttons */
    .stButton > button:disabled {
        background-color: #edf0f5 !important;
        color: #7b8495 !important;
        border-color: #d8dee8 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #e1e7f0;
        border-radius: 12px;
        padding: 0.5rem;
    }

    [data-testid="stFileUploader"] * {
        color: #172033;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #dbe4f0;
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary {
        color: #172033 !important;
    }

    /* Metrics */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #172033 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "processed" not in st.session_state:
    st.session_state.processed = False


# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def extract_pdf_text(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# --------------------------------------------------
# CREATE VECTOR STORE
# --------------------------------------------------

def create_vector_store(pdf_bytes):
    pages = extract_pdf_text(pdf_bytes)

    if not pages:
        raise ValueError(
            "No readable text found in this PDF. "
            "Scanned PDFs may require OCR."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    documents = []

    for page_data in pages:
        chunks = splitter.split_text(page_data["text"])

        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page_data["page"]},
                )
            )

    if not documents:
        raise ValueError("No text chunks could be created.")

    embeddings = load_embeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="documind_documents",
    )

    return vector_store, len(documents)


# --------------------------------------------------
# RETRIEVE RELEVANT CONTEXT
# --------------------------------------------------

def retrieve_context(question, vector_store):
    results = vector_store.similarity_search(
        question,
        k=4,
    )

    context_parts = []
    sources = []

    for document in results:
        page_number = document.metadata.get("page", "?")

        context_parts.append(
            f"[PDF Page {page_number}]\n"
            f"{document.page_content}"
        )

        sources.append({
            "page": page_number,
            "text": document.page_content,
        })

    return "\n\n".join(context_parts), sources


# --------------------------------------------------
# GENERATE ANSWER USING GROQ
# --------------------------------------------------

def generate_answer(question, context, history):
    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your local "
            ".env file or Streamlit Cloud Secrets."
        )

    client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are DocuMind, a helpful document Q&A assistant. "
                "Answer using the supplied PDF context. Do not invent "
                "facts. If the answer is not present in the context, "
                "say you could not find it in the uploaded document. "
                "Refer to PDF page numbers when useful. "
                "Keep answers clear and readable."
            ),
        }
    ]

    # Include recent conversation for follow-up questions.
    for message in history[-6:]:
        if message["role"] in ("user", "assistant"):
            messages.append({
                "role": message["role"],
                "content": message["content"],
            })

    messages.append({
        "role": "user",
        "content": (
            f"PDF context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based on the PDF context above."
        ),
    })

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
    )

    return (
        response.choices[0].message.content
        or "I couldn't generate an answer."
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">📄 DocuMind</div>',
        unsafe_allow_html=True,
    )

    st.caption("Your intelligent document assistant")
    st.divider()

    st.markdown("### 📁 Upload document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF to ask questions about it.",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.document_name:
            st.session_state.pdf_bytes = uploaded_file.getvalue()
            st.session_state.document_name = uploaded_file.name
            st.session_state.vector_store = None
            st.session_state.processed = False
            st.session_state.messages = []

    if st.session_state.pdf_bytes is not None:
        st.caption(f"Selected: {st.session_state.document_name}")

        if st.button(
            "⚙️ Process document",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Reading PDF and creating embeddings..."
                ):
                    vector_store, chunk_count = create_vector_store(
                        st.session_state.pdf_bytes
                    )

                st.session_state.vector_store = vector_store
                st.session_state.processed = True
                st.session_state.messages = []

                st.success(
                    f"Document processed! "
                    f"{chunk_count} text chunks indexed."
                )

            except Exception as error:
                st.session_state.vector_store = None
                st.session_state.processed = False
                st.error(f"Could not process PDF: {error}")

        if st.session_state.processed:
            st.success("Document ready for questions")

        # Download original PDF
        st.download_button(
            label="⬇️ Download original PDF",
            data=st.session_state.pdf_bytes,
            file_name=(
                st.session_state.document_name or "document.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )

        # Clear uploaded document
        if st.button(
            "🗑️ Clear document",
            use_container_width=True,
        ):
            st.session_state.pdf_bytes = None
            st.session_state.document_name = None
            st.session_state.vector_store = None
            st.session_state.processed = False
            st.session_state.messages = []
            st.rerun()

    st.divider()

    st.markdown("### 💡 How to use")
    st.markdown(
        """
        1. Upload a PDF.
        2. Click **Process document**.
        3. Ask questions about the PDF.
        4. Read answers and check source pages.
        """
    )

    st.caption("Powered by Groq, LangChain and Chroma")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>Chat with your documents.</h1>
        <p>
            Upload a PDF, ask questions in natural language,
            and get answers grounded in your document.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------

chat_column, document_column = st.columns(
    [2.1, 1],
    gap="large",
)


# --------------------------------------------------
# CHAT SECTION
# --------------------------------------------------

with chat_column:
    st.markdown(
        '<div class="section-title">💬 Document Q&A</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.processed:
        st.info(
            "Upload a PDF in the sidebar and click "
            "**Process document** to start asking questions."
        )

        st.markdown(
            """
            <div class="info-card">
                <h3>✨ What you can do</h3>
                <p>
                    Summarize documents, find specific information,
                    understand difficult sections, and ask follow-up
                    questions about your PDF.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        # Display previous messages.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("📚 View source passages"):
                        for source in message["sources"]:
                            st.markdown(
                                f"**PDF page {source['page']}**"
                            )
                            st.write(source["text"])

        question = st.chat_input(
            "Ask a question about your PDF..."
        )

        if question:
            previous_history = st.session_state.messages.copy()

            st.session_state.messages.append({
                "role": "user",
                "content": question,
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching your document..."):
                        context, sources = retrieve_context(
                            question,
                            st.session_state.vector_store,
                        )

                        if not context.strip():
                            answer = (
                                "I couldn't find relevant information "
                                "in the uploaded document."
                            )
                        else:
                            answer = generate_answer(
                                question,
                                context,
                                previous_history,
                            )

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 View source passages"):
                            for source in sources:
                                st.markdown(
                                    f"**PDF page {source['page']}**"
                                )
                                st.write(source["text"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as error:
                    error_message = (
                        f"Sorry, something went wrong: {error}"
                    )
                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                    })


# --------------------------------------------------
# DOCUMENT DETAILS
# --------------------------------------------------

with document_column:
    st.markdown(
        '<div class="section-title">📑 Your document</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.pdf_bytes is not None:
        document_name = st.session_state.document_name or "PDF"

        # Escape HTML characters in uploaded filename.
        safe_name = (
            document_name
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

        document_status = (
            "Ready to use"
            if st.session_state.processed
            else "Waiting to be processed"
        )

        st.markdown(
            f"""
            <div class="info-card">
                <h3>📄 {safe_name}</h3>
                <p>{document_status}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            pdf_reader = PdfReader(
                BytesIO(st.session_state.pdf_bytes)
            )

            st.metric(
                "Total PDF pages",
                len(pdf_reader.pages),
            )

            st.metric(
                "Questions asked",
                sum(
                    1
                    for message in st.session_state.messages
                    if message["role"] == "user"
                ),
            )

        except Exception:
            st.caption("PDF details are unavailable.")

    else:
        st.markdown(
            """
            <div class="info-card">
                <h3>📂 No document uploaded</h3>
                <p>
                    Upload a PDF to see document details and page
                    information here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Clear chat history
    if st.session_state.processed:
        if st.button(
            "🧹 Clear chat history",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DocuMind · RAG-based Document Question Answering"
)
