import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from citations import (  # noqa: E402
    EvalRow,
    build_citations,
    format_context,
    normalize,
    page_label,
    parse_pages,
    score_rows,
    verify_quote,
)


def doc(text, page0):
    return SimpleNamespace(page_content=text, metadata={"page": page0})


DOCS = [
    doc("The waiting period for pre-existing\nconditions is 24 months.", 3),  # page 4
    doc("Claims must be filed within 30 days of discharge.", 6),  # page 7
]


def test_page_label_is_one_based():
    assert page_label({"page": 0}) == 1
    assert page_label({}) is None


def test_normalize_collapses_whitespace_and_case():
    assert normalize("  Hello\n  WORLD ") == "hello world"


def test_verify_quote_ignores_line_breaks():
    assert verify_quote("waiting period for pre-existing conditions is 24 months", 4, DOCS)


def test_verify_quote_rejects_wrong_page():
    assert not verify_quote("Claims must be filed within 30 days of discharge.", 4, DOCS)


def test_verify_quote_rejects_invented_text():
    assert not verify_quote("Claims must be filed within 90 days of discharge.", 7, DOCS)
    assert not verify_quote("short", 7, DOCS)


def test_build_citations_dedupes_and_skips_bad_items():
    raw = [
        {"page": 4, "quote": "conditions is 24 months."},
        {"page": 4, "quote": "Conditions is 24 MONTHS."},
        {"page": "x", "quote": "bad"},
        {"quote": "no page"},
    ]
    cites = build_citations(raw, DOCS)
    assert len(cites) == 1 and cites[0].verified


def test_format_context_tags_pages():
    text = format_context(DOCS)
    assert "[Page 4]" in text and "[Page 7]" in text


def test_parse_pages():
    assert parse_pages("4") == {4}
    assert parse_pages("4|5") == {4, 5}
    assert parse_pages("4, 6") == {4, 6}
    assert parse_pages("4-6") == {4, 5, 6}
    assert parse_pages("") == set()
    assert parse_pages(None) == set()
    assert parse_pages("nan") == set()


def test_scoring_uses_verified_pages_only():
    rows = [
        EvalRow("q1", "a", {4}, cited_pages=[4], retrieved_pages=[4, 7]),
        EvalRow("q2", "a", {7}, cited_pages=[], retrieved_pages=[7]),  # retrieved, not cited
        EvalRow("q3", "a", {9}, cited_pages=[2], retrieved_pages=[2]),
        EvalRow("q4", "a", {4, 5}, cited_pages=[5], retrieved_pages=[5]),
    ]
    stats = score_rows(rows)
    assert stats["correct"] == 2 and stats["total"] == 4
    assert stats["accuracy"] == 0.5
    assert stats["retrieval_hits"] == 3


def test_empty_scoring():
    assert score_rows([])["accuracy"] == 0.0
