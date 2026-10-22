# 📚 DocuMind — RAG Document Q&A

An AI-powered document question-answering application that allows users to upload PDF documents and ask questions about their content. It retrieves relevant passages and generates answers grounded in the uploaded document, with source page citations.

## 🚀 Features

- **PDF Upload:** Upload and process PDF documents.
- **Text Extraction:** Extract readable text from PDF pages using PyPDF.
- **Text Chunking:** Split documents into manageable chunks using RecursiveCharacterTextSplitter.
- **Embeddings:** Convert text chunks into vector embeddings using Sentence Transformers.
- **Vector Database:** Store and retrieve document chunks using ChromaDB.
- **Semantic Retrieval:** Retrieve relevant passages for user questions.
- **AI-Generated Answers:** Generate answers using the Groq API.
- **Source Citations:** Display supporting PDF page numbers and retrieved passages.
- **Interactive Dashboard:** Use a Streamlit interface to chat with your documents and preview the PDF.
- **Grounded Responses:** Instruct the model to acknowledge when the retrieved content does not contain enough information.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Web interface |
| LangChain | Text splitting and retrieval workflow |
| PyPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector storage and similarity search |
| Groq API | Language model responses |
| python-dotenv | Environment variable management |

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Task-2-RAG-Document-QA
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install streamlit pypdf chromadb langchain langchain-text-splitters langchain-chroma langchain-huggingface sentence-transformers python-dotenv groq
```

### 4. Configure the API key

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own API key. Never commit your API key to GitHub.

### 5. Run the application

```bash
streamlit run app.py --server.fileWatcherType none
```

Open the local URL displayed in the terminal.

## 📖 How It Works

1. The user uploads a PDF document.
2. The application extracts text from each page.
3. RecursiveCharacterTextSplitter divides the text into overlapping chunks.
4. Sentence Transformers generates embeddings for the chunks.
5. ChromaDB stores the embeddings and associated page metadata.
6. When a question is submitted, the application retrieves relevant passages through similarity search.
7. The retrieved passages are provided to the Groq language model as context.
8. The model generates an answer based on the supplied excerpts, with source page references.
9. Users can inspect the retrieved passages to verify the answer.

## 🧪 Example

**Question:** What are eigenvalues?

**Expected result:** A definition based on the uploaded mathematics PDF, with a citation to the relevant PDF page.

**Out-of-document question:** If the retrieved excerpts do not contain the requested information, the application is instructed to say that the document does not provide enough information.

## 🔐 Security

- Store API credentials in `.env`.
- Add `.env` and `.venv/` to `.gitignore`.
- Never publish API keys or other secrets.
- Do not upload confidential documents to services unless authorized.

## ⚠️ Limitations

- Text extraction quality depends on the PDF structure.
- Scanned PDFs may require OCR support.
- Answers depend on the relevance and quality of retrieved passages.
- Page citations refer to the original PDF page numbering.
- The language model may still make mistakes; verify answers against the source document.

## 🎯 Project Objective

To demonstrate a Retrieval-Augmented Generation (RAG) pipeline that combines document processing, embeddings, vector similarity search, and language-model-based question answering with source citations.

---

**Project:** DocuMind — RAG Document Q&A  
**Task:** Task 2 — Document Question Answering
