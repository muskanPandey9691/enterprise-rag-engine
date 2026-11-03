"""RAG pipeline: PDF -> page-tagged chunks -> retrieval -> answer with citations."""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from citations import (
    Citation,
    EvalRow,
    build_citations,
    format_context,
    page_label,
    parse_pages,
)

DEFAULT_LLM_MODEL = "gemini-3.8-flash"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


# --------------------------------------------------------------------------- #
# Structured output the LLM must return
# --------------------------------------------------------------------------- #
class ClaimedCitation(BaseModel):
    page: int = Field(description="Page number taken from the [Page N] tag of the context chunk.")
    quote: str = Field(
        description="One exact sentence or line copied character-for-character from that page's "
        "text in the context. No paraphrasing, no ellipses."
    )


class CitedAnswer(BaseModel):
    answer: str = Field(description="Concise answer grounded only in the context.")
    citations: list[ClaimedCitation] = Field(
        default_factory=list,
        description="1-3 citations supporting the answer. Empty if the context has no answer.",
    )


SYSTEM_PROMPT = (
    "You are a compliance document assistant. Answer using ONLY the supplied context. "
    "If the context does not contain the answer, say that the information is not available "
    "in the uploaded document and return no citations. Never invent facts.\n\n"
    "Every context chunk starts with a [Page N] tag. For each fact you use, add a citation "
    "with that page number and the EXACT sentence or line from the context that supports it, "
    "copied character-for-character. Do not paraphrase the quote.\n\n"
    "Context:\n{context}"
)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class IndexedDocument:
    name: str
    vector_store: Chroma
    page_count: int
    chunk_count: int


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieved_docs: list = field(default_factory=list)
    structured: bool = True  # False if the model's JSON could not be parsed

    @property
    def retrieved_pages(self) -> list[int]:
        pages = [page_label(d.metadata) for d in self.retrieved_docs]
        return sorted({p for p in pages if p is not None})

    @property
    def verified_pages(self) -> list[int]:
        return sorted({c.page for c in self.citations if c.verified})


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_models(api_key: str, llm_model: str = DEFAULT_LLM_MODEL):
    """Create the embedding model and the chat model (imported lazily)."""
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    llm = ChatGoogleGenerativeAI(model=llm_model, max_retries=6)
    return embeddings, llm


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def build_index(
    pdf_bytes: bytes,
    name: str,
    embeddings,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> IndexedDocument:
    """Load a PDF page by page and index it. Each chunk keeps `page` and `source`
    metadata, which is what lets every answer point back to its source page."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
            handle.write(pdf_bytes)
            temp_path = handle.name
        pages = PyPDFLoader(temp_path).load()
    finally:
        if temp_path and Path(temp_path).exists():
            os.unlink(temp_path)

    if not pages or not any(p.page_content.strip() for p in pages):
        raise ValueError("No readable text was found. This may be a scanned PDF; run OCR first.")

    for page in pages:
        page.metadata["source"] = name  # PyPDFLoader would otherwise expose a temp path

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("The PDF did not contain any text chunks to index.")

    # A unique collection per index: Chroma's in-memory client is shared across a
    # process, so re-using the default collection would mix chunks from old PDFs.
    vector_store = Chroma.from_documents(
        chunks, embeddings, collection_name=f"docuquery-{uuid.uuid4().hex}"
    )
    return IndexedDocument(
        name=name, vector_store=vector_store, page_count=len(pages), chunk_count=len(chunks)
    )


# --------------------------------------------------------------------------- #
# Question answering with citations
# --------------------------------------------------------------------------- #
def answer_question(index: IndexedDocument, llm, question: str, top_k: int = 4) -> AnswerResult:
    docs = index.vector_store.similarity_search(question, k=top_k)
    if not docs:
        return AnswerResult(answer="No relevant passages were found in the document.")

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
    inputs = {"context": format_context(docs), "question": question}

    try:
        parsed: CitedAnswer | None = (prompt | llm.with_structured_output(CitedAnswer)).invoke(inputs)
    except (OutputParserException, ValueError):
        parsed = None  # fall back below; API/auth/quota errors are NOT caught here

    if parsed is None:
        message = (prompt | llm).invoke(inputs)
        text = message.content if isinstance(message.content, str) else str(message.content)
        return AnswerResult(answer=text, citations=[], retrieved_docs=docs, structured=False)

    citations = build_citations([c.model_dump() for c in parsed.citations], docs)
    return AnswerResult(answer=parsed.answer, citations=citations, retrieved_docs=docs)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(index: IndexedDocument, llm, questions: list[dict], top_k: int = 4, progress=None) -> list[EvalRow]:
    """`questions` is a list of {"question": str, "expected_page": "4" | "4|5" | ...}."""
    rows: list[EvalRow] = []
    for i, item in enumerate(questions):
        question = str(item.get("question", "")).strip()
        expected = parse_pages(item.get("expected_page"))
        if not question:
            continue
        try:
            result = answer_question(index, llm, question, top_k=top_k)
            row = EvalRow(
                question=question,
                answer=result.answer,
                expected_pages=expected,
                cited_pages=result.verified_pages,
                retrieved_pages=result.retrieved_pages,
            )
        except Exception as exc:  # keep going; one failed question must not kill the run
            row = EvalRow(question=question, answer="", expected_pages=expected, error=str(exc)[:200])
        rows.append(row)
        if progress:
            progress(i + 1, len(questions))
    return rows
