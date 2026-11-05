import os
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import AgentToolbox, run_agent, safe_eval  # noqa: E402
from rag_core import CitedAnswer, ClaimedCitation  # noqa: E402


def doc(text, page0):
    return SimpleNamespace(page_content=text, metadata={"page": page0})


class FakeStore:
    def __init__(self, docs, pages=None):
        self.docs = docs
        self.pages = pages or {}
        self.queries = []

    def similarity_search(self, query, k=4):
        self.queries.append(query)
        return self.docs[:k]

    def get(self, where=None):
        texts = self.pages.get(where["page"], [])
        return {"documents": texts, "metadatas": [{"page": where["page"]} for _ in texts]}


def make_index(docs, pages=None):
    return SimpleNamespace(vector_store=FakeStore(docs, pages), page_count=10)


class ScriptedLLM:
    """Replays scripted agent turns, then answers the final structured call."""

    def __init__(self, script, cited):
        self.script = list(script)
        self.cited = cited
        self.seen_prompts = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.script.pop(0) if self.script else AIMessage(content="DONE")

    def with_structured_output(self, schema):
        def respond(prompt_value):
            self.seen_prompts.append(prompt_value.to_string())
            return self.cited

        return RunnableLambda(respond)


def tool_call(name, args, call_id="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


# ----------------------------- calculator ---------------------------------- #
@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("12 * 3 * (1 + 5 / 100)", 37.8),
        ("-5 + 2", -3),
        ("7 // 2", 3),
        ("2 ** 10", 1024),
    ],
)
def test_safe_eval_arithmetic(expression, expected):
    assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo hi')", "open('x')", "a + 1", "2 +", "", "2 ** 1000", "(1).real", "'a' * 3", "x" * 300],
)
def test_safe_eval_rejects_anything_but_arithmetic(expression):
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_safe_eval_division_by_zero_is_a_zero_division_error():
    with pytest.raises(ZeroDivisionError):
        safe_eval("1 / 0")


# ------------------------------- tools ------------------------------------- #
def test_calculate_tool_records_results_and_reports_errors():
    box = AgentToolbox(make_index([]))
    assert box.by_name["calculate"].invoke({"expression": "10 * 3"}) == "30"
    assert box.calculations == ["10 * 3 = 30"]
    assert box.by_name["calculate"].invoke({"expression": "1 / 0"}).startswith("Calculator error")
    assert box.by_name["calculate"].invoke({"expression": "os.system('x')"}).startswith("Calculator error")
    assert len(box.calculations) == 1


def test_search_tool_tags_pages_and_remembers_evidence():
    box = AgentToolbox(make_index([doc("Claims must be filed within 30 days.", 6)]))
    out = box.by_name["search_document"].invoke({"query": "claim deadline"})
    assert out.startswith("[Page 7]")
    assert len(box.docs) == 1


def test_get_page_tool_validates_the_page_number():
    box = AgentToolbox(make_index([], pages={4: ["Premium is 500 per month."]}))
    get_page = box.by_name["get_page"]
    assert get_page.invoke({"page_number": 99}) == "The document has pages 1 to 10."
    assert get_page.invoke({"page_number": 3}) == "No text was indexed for page 3."
    assert get_page.invoke({"page_number": 5}) == "[Page 5]\nPremium is 500 per month."
    assert box.docs[0].metadata["page"] == 4


# ------------------------------- agent loop --------------------------------- #
def test_agent_uses_tools_then_gives_a_verified_answer_with_calculations():
    docs = [doc("The monthly premium is 500.", 1)]  # page 2
    cited = CitedAnswer(
        answer="Annual premium is 6000.",
        citations=[ClaimedCitation(page=2, quote="The monthly premium is 500.")],
    )
    llm = ScriptedLLM(
        [
            tool_call("search_document", {"query": "monthly premium"}, "a"),
            tool_call("calculate", {"expression": "500 * 12"}, "b"),
            AIMessage(content="DONE"),
        ],
        cited,
    )

    run = run_agent(make_index(docs), llm, "What is the annual premium?")

    assert run.steps == ["search_document(query='monthly premium')", "calculate(expression='500 * 12')"]
    assert run.answer.answer == "Annual premium is 6000."
    assert [c.verified for c in run.answer.citations] == [True]
    assert "500 * 12 = 6000" in llm.seen_prompts[0]  # calculator result reached the final answer step


def test_agent_falls_back_to_plain_retrieval_if_it_never_searches():
    docs = [doc("The monthly premium is 500.", 1)]
    cited = CitedAnswer(answer="500.", citations=[ClaimedCitation(page=2, quote="The monthly premium is 500.")])
    llm = ScriptedLLM([AIMessage(content="I already know.")], cited)
    index = make_index(docs)

    run = run_agent(index, llm, "What is the premium?")

    assert index.vector_store.queries == ["What is the premium?"]
    assert run.steps[-1].endswith("[fallback]")
    assert run.answer.citations[0].verified


def test_agent_stops_after_max_steps():
    docs = [doc("Some text here for the test.", 0)]
    cited = CitedAnswer(answer="x", citations=[])
    llm = ScriptedLLM([tool_call("search_document", {"query": "q"}, str(i)) for i in range(20)], cited)

    run = run_agent(make_index(docs), llm, "q", max_steps=3)

    assert len(run.steps) == 3


def test_agent_survives_unknown_tools_and_bad_arguments():
    docs = [doc("The monthly premium is 500.", 1)]
    cited = CitedAnswer(answer="ok", citations=[])
    llm = ScriptedLLM(
        [tool_call("delete_everything", {"x": 1}, "a"), tool_call("get_page", {"page_number": "abc"}, "b")],
        cited,
    )

    run = run_agent(make_index(docs), llm, "q")

    assert [s.split("(")[0] for s in run.steps][:2] == ["delete_everything", "get_page"]
    assert run.answer.answer == "ok"
