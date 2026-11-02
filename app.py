import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="Enterprise RAG Engine",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Enterprise RAG Engine")
st.caption("Ask questions about your PDFs and receive answers grounded in retrieved document content.")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Google Gemini API key",
        type="password",
        help="Your key is used only for this app session. Do not commit it to GitHub.",
    )
    st.markdown("[Get a Gemini API key](https://aistudio.google.com/apikey)")
    st.divider()
    st.subheader("Retrieval settings")
    chunk_size = st.slider("Chunk size (characters)", 300, 1500, 700, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 300, 100, step=25)
    top_k = st.slider("Retrieved chunks", 1, 8, 4)

if not api_key:
    st.info("Add your Google Gemini API key in the sidebar to get started.")
    st.markdown(
        """
        **How it works**
        1. Add your Gemini API key.
        2. Upload a PDF document.
        3. Wait while the app extracts and indexes its text.
        4. Ask questions and inspect the source passages used in each answer.
        """
    )
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


@st.cache_resource(show_spinner=False)
def load_models(_api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    max_retries=6
)
    return embeddings, llm


try:
    embeddings, llm = load_models(api_key)
except Exception as exc:
    st.error(f"Could not initialize Gemini. Check your API key and model access. Details: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF to create your searchable knowledge base.")
    st.stop()

# A new upload or changed settings should create a fresh index.
file_signature = (uploaded_file.name, uploaded_file.size, chunk_size, chunk_overlap)
if st.session_state.get("file_signature") != file_signature:
    with st.spinner("Reading PDF and building the searchable index..."):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            pages = PyPDFLoader(temp_path).load()
            if not pages or not any(page.page_content.strip() for page in pages):
                st.error("No readable text was found. This may be a scanned PDF; run OCR first.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            chunks = splitter.split_documents(pages)
            if not chunks:
                st.error("The PDF did not contain any text chunks to index.")
                st.stop()

            vector_store = Chroma.from_documents(chunks, embeddings)
            st.session_state.vector_store = vector_store
            st.session_state.file_signature = file_signature
            st.session_state.document_name = uploaded_file.name
            st.session_state.chunk_count = len(chunks)
            st.session_state.page_count = len(pages)
        except Exception as exc:
            st.error(f"Document processing failed: {exc}")
            st.stop()
        finally:
            if temp_path and Path(temp_path).exists():
                os.unlink(temp_path)

st.success(
    f"Indexed **{st.session_state.document_name}** — "
    f"{st.session_state.page_count} pages, {st.session_state.chunk_count} text chunks."
)

retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": top_k})

system_prompt = (
    "You are an enterprise document assistant. Answer using only the supplied context. "
    "If the context does not contain the answer, clearly say that the information is "
    "not available in the uploaded document. Do not invent facts. "
    "When useful, mention the page number(s) supporting your answer.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

st.divider()
st.subheader("💬 Ask your document")
question = st.text_input(
    "Enter a question",
    placeholder="For example: What are the main objectives described in this document?",
    key="question",
)

if question:
    with st.spinner("Searching the document and preparing an answer..."):
        try:
            result = rag_chain.invoke({"input": question})
            st.markdown("### Answer")
            st.write(result["answer"])

            with st.expander("📖 View retrieved source passages", expanded=True):
                for index, doc in enumerate(result.get("context", []), start=1):
                    page_number = doc.metadata.get("page")
                    page_label = page_number + 1 if isinstance(page_number, int) else "Unknown"
                    st.markdown(f"**Source {index} · Page {page_label}**")
                    st.caption(doc.page_content)
                    if index < len(result.get("context", [])):
                        st.divider()
        except Exception as exc:
            st.error(f"Question answering failed: {exc}")

st.caption("Note: Answers are generated by an LLM. Always verify important information against the cited source passages.")
