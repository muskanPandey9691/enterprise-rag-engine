"""Pure helpers for source citation and evaluation.

No LangChain or network imports here, so everything in this file can be
unit-tested without an API key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower-case and collapse whitespace so PDF line breaks don't break matching."""
    text = text.replace("­", "")  # soft hyphens from PDF extraction
    return _WS.sub(" ", text).strip().lower()


@dataclass
class Citation:
    page: int  # 1-based page number, as a human sees it in the PDF viewer
    quote: str  # the exact line / sentence the answer relies on
    verified: bool = False  # True if the quote was found verbatim on that page


def page_label(metadata: dict) -> int | None:
    """PyPDFLoader stores a 0-based page index; convert to a 1-based page number."""
    page = metadata.get("page")
    return page + 1 if isinstance(page, int) else None


def format_context(docs: Sequence) -> str:
    """Label every retrieved chunk with its page so the LLM can cite it."""
    blocks = []
    for doc in docs:
        label = page_label(doc.metadata)
        blocks.append(f"[Page {label if label is not None else '?'}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def verify_quote(quote: str, page: int, docs: Iterable) -> bool:
    """True only if `quote` appears verbatim (ignoring whitespace/case) in a
    retrieved chunk that came from `page`. This is the audit-trail check: it
    stops the model from citing a sentence that is not really in the document."""
    needle = normalize(quote)
    if len(needle) < 8:
        return False
    for doc in docs:
        if page_label(doc.metadata) == page and needle in normalize(doc.page_content):
            return True
    return False


def build_citations(raw_citations: Iterable[dict], docs: Sequence) -> list[Citation]:
    """Turn the model's claimed citations into verified Citation objects."""
    seen: set[tuple[int, str]] = set()
    out: list[Citation] = []
    for item in raw_citations:
        try:
            page = int(item["page"])
            quote = str(item["quote"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        key = (page, normalize(quote))
        if not quote or key in seen:
            continue
        seen.add(key)
        out.append(Citation(page=page, quote=quote, verified=verify_quote(quote, page, docs)))
    return out


def parse_pages(value) -> set[int]:
    """Parse an expected-page cell such as '4', '4|5', '4, 5' or '4-6' into a set."""
    pages: set[int] = set()
    if value is None:
        return pages
    for part in re.split(r"[|,;/\s]+", str(value).strip()):
        if not part:
            continue
        if re.fullmatch(r"\d+-\d+", part):
            lo, hi = (int(x) for x in part.split("-"))
            if lo <= hi and hi - lo <= 50:
                pages.update(range(lo, hi + 1))
        elif part.isdigit():
            pages.add(int(part))
    return pages


@dataclass
class EvalRow:
    question: str
    answer: str
    expected_pages: set[int]
    cited_pages: list[int] = field(default_factory=list)
    retrieved_pages: list[int] = field(default_factory=list)
    error: str = ""

    @property
    def correct(self) -> bool:
        """Correct = the bot cited a *verified* page that matches the expected page."""
        return bool(self.expected_pages & set(self.cited_pages))

    @property
    def retrieved(self) -> bool:
        return bool(self.expected_pages & set(self.retrieved_pages))


def score_rows(rows: Sequence[EvalRow]) -> dict:
    total = len(rows)
    correct = sum(r.correct for r in rows)
    retrieved = sum(r.retrieved for r in rows)
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "retrieval_hits": retrieved,
        "retrieval_rate": (retrieved / total) if total else 0.0,
    }


def rows_to_records(rows: Sequence[EvalRow]) -> list[dict]:
    """Flat records for a table / CSV: Question | Bot Answer | Expected Page | ..."""
    records = []
    for r in rows:
        records.append(
            {
                "Question": r.question,
                "Bot Answer": r.answer if not r.error else f"ERROR: {r.error}",
                "Expected Page": ", ".join(str(p) for p in sorted(r.expected_pages)),
                "Cited Page (verified)": ", ".join(str(p) for p in r.cited_pages) or "-",
                "Correct?": "✅" if r.correct else "❌",
            }
        )
    return records
