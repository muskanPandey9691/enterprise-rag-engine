import os
import sys
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chat import (  # noqa: E402
    ChatTurn,
    chat_answer,
    condense_question,
    format_history,
    message_text,
)
from rag_core import CitedAnswer, ClaimedCitation  # noqa: E402


def doc(text, page0):
    return SimpleNamespace(page_content=text, metadata={"page": page0})


class FakeStore:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def similarity_search(self, query, k=4):
        self.queries.append(query)
        return self.docs[:k]


class FakeLLM:
    """Plain `invoke` returns a canned rewrite; structured output returns a canned answer."""

    def __init__(self, rewrite="", cited=None):
        self.rewrite = rewrite
        self.cited = cited
        self.invoke_calls = 0

    def invoke(self, messages):
        self.invoke_calls += 1
        return SimpleNamespace(content=self.rewrite)

    def with_structured_output(self, schema):
        return RunnableLambda(lambda _prompt: self.cited)


def make_index(docs):
    return SimpleNamespace(vector_store=FakeStore(docs), page_count=10)


HISTORY = [
    ChatTurn("user", "What is the waiting period for pre-existing conditions?"),
    ChatTurn("assistant", "It is 24 months."),
]


def test_message_text_handles_string_and_parts():
    assert message_text("hello") == "hello"
    assert message_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"


def test_format_history_labels_and_limits_turns():
    turns = [ChatTurn("user", f"q{i}") for i in range(10)]
    text = format_history(turns, max_turns=3)
    assert text.splitlines() == ["User: q7", "User: q8", "User: q9"]


def test_format_history_truncates_long_turns():
    text = format_history([ChatTurn("assistant", "word " * 500)])
    assert text.startswith("Assistant: ") and text.endswith("...")
    assert len(text) < 700


def test_condense_without_history_does_not_call_the_model():
    llm = FakeLLM(rewrite="should not be used")
    assert condense_question(llm, [], "What is covered?") == "What is covered?"
    assert llm.invoke_calls == 0


def test_condense_uses_the_rewrite():
    llm = FakeLLM(rewrite="What is the waiting period for dependants?")
    out = condense_question(llm, HISTORY, "And for dependants?")
    assert out == "What is the waiting period for dependants?"
    assert llm.invoke_calls == 1


def test_condense_ignores_empty_or_runaway_rewrites():
    assert condense_question(FakeLLM(rewrite="   "), HISTORY, "And for dependants?") == "And for dependants?"
    assert condense_question(FakeLLM(rewrite="x" * 800), HISTORY, "And for dependants?") == "And for dependants?"


def test_chat_answer_searches_the_standalone_question_and_verifies_citations():
    docs = [doc("The waiting period for dependants is 12 months.", 4)]  # page 5
    cited = CitedAnswer(
        answer="12 months.",
        citations=[
            ClaimedCitation(page=5, quote="The waiting period for dependants is 12 months."),
            ClaimedCitation(page=5, quote="A line the model made up entirely."),
        ],
    )
    llm = FakeLLM(rewrite="What is the waiting period for dependants?", cited=cited)
    index = make_index(docs)

    reply = chat_answer(index, llm, "And for dependants?", HISTORY, top_k=4)

    assert index.vector_store.queries == ["What is the waiting period for dependants?"]
    assert reply.standalone_question == "What is the waiting period for dependants?"
    assert reply.answer.answer == "12 months."
    assert [c.verified for c in reply.answer.citations] == [True, False]
    assert reply.steps == []
