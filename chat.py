"""Multi-turn chat on top of the RAG pipeline.

A follow-up such as "What about for dependants?" cannot be searched as it is: the
vector store has no idea what "what about" refers to. So every new message is first
rewritten into a standalone question using the recent conversation (conversation
memory), and only then retrieved and answered. Citations are still verified against
the retrieved passages, exactly like the single-question flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate

from rag_core import AnswerResult, IndexedDocument, answer_question

MAX_HISTORY_TURNS = 6  # how many recent messages the model gets to see
MAX_TURN_CHARS = 600  # long answers are shortened before they go into the rewrite prompt
MAX_STANDALONE_CHARS = 500  # a longer "rewrite" is probably the model answering, so we ignore it

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest message as ONE standalone question that can be understood "
            "without the conversation. Resolve references such as 'it', 'that clause' or 'the second "
            "one' using the conversation. Do not answer the question and do not add facts. If the "
            "message is already standalone, return it unchanged. Return only the question text.",
        ),
        (
            "human",
            "Conversation so far:\n{history}\n\nLatest message: {question}\n\nStandalone question:",
        ),
    ]
)


@dataclass
class ChatReply:
    """Everything the UI needs to show one assistant turn."""

    answer: AnswerResult
    standalone_question: str  # what was actually searched
    steps: list[str] = field(default_factory=list)  # agent tool calls (empty in plain mode)


@dataclass
class ChatTurn:
    role: str  # "user" or "assistant"
    content: str
    reply: ChatReply | None = None  # set on assistant turns so citations can be shown again


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without an API key)
# --------------------------------------------------------------------------- #
def message_text(content) -> str:
    """Gemini sometimes returns a list of parts instead of a string; flatten it."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return "".join(parts)
    return str(content)


def format_history(turns: list[ChatTurn], max_turns: int = MAX_HISTORY_TURNS) -> str:
    lines = []
    for turn in turns[-max_turns:]:
        who = "User" if turn.role == "user" else "Assistant"
        text = " ".join(turn.content.split())
        if len(text) > MAX_TURN_CHARS:
            text = text[:MAX_TURN_CHARS] + "..."
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def condense_question(llm, history: list[ChatTurn], question: str) -> str:
    """Turn a follow-up into a standalone question. With no history, nothing to do."""
    if not history:
        return question
    messages = CONDENSE_PROMPT.format_messages(history=format_history(history), question=question)
    rewritten = " ".join(message_text(llm.invoke(messages).content).split())
    if not rewritten or len(rewritten) > MAX_STANDALONE_CHARS:
        return question  # never let a bad rewrite replace what the user actually asked
    return rewritten


# --------------------------------------------------------------------------- #
# Entry point used by the app
# --------------------------------------------------------------------------- #
def chat_answer(
    index: IndexedDocument,
    llm,
    question: str,
    history: list[ChatTurn],
    top_k: int = 4,
    use_agent: bool = False,
) -> ChatReply:
    standalone = condense_question(llm, history, question)
    if use_agent:
        from agent import run_agent  # imported here so plain chat never needs the tool stack

        run = run_agent(index, llm, standalone, top_k=top_k)
        return ChatReply(answer=run.answer, standalone_question=standalone, steps=run.steps)
    result = answer_question(index, llm, standalone, top_k=top_k)
    return ChatReply(answer=result, standalone_question=standalone)
