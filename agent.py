"""A small tool-using agent over the indexed document.

Instead of one fixed "retrieve once, then answer" step, the model decides what to do:

    search_document(query)  - look for passages (can be called several times with different queries)
    get_page(page_number)   - read one page when the user names it
    calculate(expression)   - do arithmetic exactly instead of guessing numbers

The loop: the model asks for tools -> we run them -> results go back to the model -> repeat
until it stops asking (or MAX_STEPS is reached). Then comes the important part: the final
answer is NOT free text from the agent. It goes through the same grounded, citation-verified
step as the normal Q&A (`generate_cited_answer`), using only the passages the tools found.
So the agent adds flexibility without losing the audit trail.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from citations import format_context, page_label
from rag_core import AnswerResult, IndexedDocument, generate_cited_answer

MAX_STEPS = 5
MAX_PAGE_CHARS = 3000

AGENT_SYSTEM_PROMPT = (
    "You are a research assistant for ONE uploaded document. Gather the facts needed to answer "
    "the user's question using your tools: call search_document (several times with different "
    "queries if needed), get_page when the user names a page, and calculate for any arithmetic. "
    "Never answer from memory. When you have enough evidence, reply with just: DONE"
)


# --------------------------------------------------------------------------- #
# Safe calculator: parses the expression instead of running eval()
# --------------------------------------------------------------------------- #
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 100


def safe_eval(expression: str) -> float:
    """Evaluate plain arithmetic. Anything else (names, calls, attributes) is rejected."""
    expression = expression.strip()
    if not expression or len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError("expression is empty or too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("not a valid arithmetic expression") from exc

    def walk(node, depth=0):
        if depth > 20:
            raise ValueError("expression is nested too deeply")
        if isinstance(node, ast.Expression):
            return walk(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = walk(node.left, depth + 1), walk(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ValueError("exponent is too large")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](walk(node.operand, depth + 1))
        raise ValueError("only numbers and + - * / // % ** ( ) are allowed")

    return walk(tree)


# --------------------------------------------------------------------------- #
# Tools (each one remembers what it found so the final answer can be verified)
# --------------------------------------------------------------------------- #
class AgentToolbox:
    def __init__(self, index: IndexedDocument, top_k: int = 4):
        self.index = index
        self.top_k = top_k
        self.docs: list = []  # every passage any tool returned
        self.calculations: list[str] = []  # "expression = value" lines

        @tool
        def search_document(query: str) -> str:
            """Search the uploaded document for passages relevant to the query.
            Returns passages, each starting with a [Page N] tag."""
            found = self.index.vector_store.similarity_search(query, k=self.top_k)
            self.docs.extend(found)
            return format_context(found) if found else "No relevant passages found."

        @tool
        def get_page(page_number: int) -> str:
            """Return the text of one page of the document. page_number is 1-based, as shown
            in a PDF viewer."""
            if not 1 <= page_number <= self.index.page_count:
                return f"The document has pages 1 to {self.index.page_count}."
            data = self.index.vector_store.get(where={"page": page_number - 1})
            texts = data.get("documents") or []
            metas = data.get("metadatas") or []
            if not texts:
                return f"No text was indexed for page {page_number}."
            self.docs.extend(Document(page_content=t, metadata=m) for t, m in zip(texts, metas))
            body = "\n".join(texts)[:MAX_PAGE_CHARS]
            return f"[Page {page_number}]\n{body}"

        @tool
        def calculate(expression: str) -> str:
            """Evaluate an arithmetic expression such as '12 * 3 * (1 + 5 / 100)'. Write
            percentages as fractions (5 / 100). Use numbers taken from the document."""
            try:
                value = safe_eval(expression)
            except (ValueError, ZeroDivisionError, OverflowError) as exc:
                return f"Calculator error: {exc}"
            self.calculations.append(f"{expression.strip()} = {value}")
            return str(value)

        self.tools = [search_document, get_page, calculate]
        self.by_name = {t.name: t for t in self.tools}


@dataclass
class AgentRun:
    answer: AnswerResult
    steps: list[str] = field(default_factory=list)


def _dedupe(docs: list) -> list:
    seen, unique = set(), []
    for doc in docs:
        key = (page_label(doc.metadata), doc.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def run_agent(
    index: IndexedDocument, llm, question: str, top_k: int = 4, max_steps: int = MAX_STEPS
) -> AgentRun:
    box = AgentToolbox(index, top_k=top_k)
    tool_llm = llm.bind_tools(box.tools)
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), HumanMessage(content=question)]
    steps: list[str] = []

    for _ in range(max_steps):
        ai = tool_llm.invoke(messages)
        messages.append(ai)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            break  # the model says it has enough evidence
        for call in calls:
            tool_fn = box.by_name.get(call["name"])
            if tool_fn is None:
                output = f"Unknown tool: {call['name']}"
            else:
                try:
                    output = tool_fn.invoke(call["args"])
                except Exception as exc:  # a bad tool call must not crash the whole answer
                    output = f"Tool error: {exc}"
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            steps.append(f"{call['name']}({args})")
            messages.append(ToolMessage(content=str(output), tool_call_id=call.get("id") or call["name"]))

    docs = _dedupe(box.docs)
    if not docs:  # the agent never searched, so fall back to one plain retrieval
        docs = index.vector_store.similarity_search(question, k=top_k)
        steps.append(f"search_document(query={question!r})  [fallback]")
    if not docs:
        return AgentRun(AnswerResult(answer="No relevant passages were found in the document."), steps)

    extra = ""
    if box.calculations:
        extra = "Calculator results (computed from numbers in the document):\n" + "\n".join(box.calculations)
    return AgentRun(generate_cited_answer(llm, question, docs, extra_context=extra), steps)
