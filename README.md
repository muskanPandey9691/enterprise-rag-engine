# Enterprise RAG Engine

A Streamlit application that lets you upload a PDF, index its contents with embeddings, and ask questions using Retrieval-Augmented Generation (RAG). Answers are instructed to stay grounded in the retrieved document context, and the app displays the source passages and page numbers.

## Features

- Upload and process PDF documents
- Split document text into configurable overlapping chunks
- Create a Chroma vector index using Google Gemini embeddings
- Retrieve the most relevant chunks for a question
- Generate context-grounded answers with Gemini
- Inspect retrieved passages and their source page numbers

## Tech stack

- Python + Streamlit
- LangChain
- ChromaDB
- Google Gemini API
- PyPDF

## Run locally

### 1. Download or clone this repository

```bash
git clone https://github.com/YOUR-USERNAME/enterprise-rag-engine.git
cd enterprise-rag-engine
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`. Paste your Google Gemini API key into the sidebar, upload a PDF, and start asking questions.

Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Select **Create app**, choose your repository and the `app.py` file.
4. Deploy. Enter your Gemini API key in the app's sidebar when it loads.

Do not commit API keys, credentials, private PDFs, or local database files to GitHub.

## Project structure

```text
enterprise-rag-engine/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Important notes

- This starter version processes one uploaded PDF at a time. Uploading a different PDF or changing chunk settings rebuilds the index.
- The PDF must contain selectable text. Scanned/image-only PDFs need OCR before upload.
- The app displays retrieved source passages to make answers easier to verify. Retrieval and LLM answers can still be incomplete or incorrect; check important claims against the original PDF.
- The app uses the Gemini API. API availability and model names can change; if Google changes model access, update the model identifiers in `app.py`.

## License

Add a license file if you plan to publish or distribute this project.
