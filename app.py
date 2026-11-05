import pandas as pd
import streamlit as st

from chat import ChatReply, ChatTurn, chat_answer
from citations import page_label, rows_to_records, score_rows
from rag_core import DEFAULT_LLM_MODEL, answer_question, build_index, evaluate, make_models

APP_NAME = "DocuQuery"
APP_TAGLINE = "RAG-based PDF Compliance Assistant"

st.set_page_config(page_title=f"{APP_NAME} — {APP_TAGLINE}", page_icon="📚", layout="wide")

st.title(f"📚 {APP_NAME}")
st.caption(
    f"{APP_TAGLINE} · Every answer comes with the page number and the exact line it is based on, "
    "so it can be audited."
)

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Google Gemini API key",
        type="password",
        help="Your key is used only for this app session. Do not commit it to GitHub.",
    )
    st.markdown("[Get a Gemini API key](https://aistudio.google.com/apikey)")
    llm_model = st.text_input("Gemini model", value=DEFAULT_LLM_MODEL)
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
        3. Ask questions. Each answer cites the **page** and the **exact line** it relies on,
           and the app checks that the line really exists on that page.
        4. Use the **Evaluate** tab to measure accuracy on your own test questions.
        """
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def load_models(_api_key: str, model_name: str):
    return make_models(_api_key, model_name)


try:
    embeddings, llm = load_models(api_key, llm_model)
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
        try:
            st.session_state.index = build_index(
                uploaded_file.getvalue(),
                uploaded_file.name,
                embeddings,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.session_state.file_signature = file_signature
            st.session_state.answer_cache = {}
            st.session_state.chat_history = []  # a new document means a new conversation
            st.session_state.eval_rows = None
        except Exception as exc:
            st.error(f"Document processing failed: {exc}")
            st.stop()

index = st.session_state.index
st.success(f"Indexed **{index.name}** — {index.page_count} pages, {index.chunk_count} text chunks.")

ask_tab, chat_tab, eval_tab = st.tabs(["💬 Ask", "🗨️ Chat", "🧪 Evaluate"])


def render_reply(reply: ChatReply) -> None:
    """Show one assistant turn: answer, verified sources, and what the agent did."""
    st.write(reply.answer.answer)
    if reply.answer.citations:
        for n, cite in enumerate(reply.answer.citations, start=1):
            badge = "✅ verified in document" if cite.verified else "⚠️ not found verbatim on this page"
            st.markdown(f"**[{n}] Page {cite.page}** · {badge}")
            st.markdown(f"> {cite.quote}")
        if any(not c.verified for c in reply.answer.citations):
            st.warning("At least one cited line could not be matched word-for-word. Check it before relying on this answer.")
    elif reply.answer.structured:
        st.caption("No supporting source was cited — the document may not contain this answer.")
    else:
        st.warning("The model did not return structured citations for this answer.")
    if reply.standalone_question:
        st.caption(f"🔍 Searched for: {reply.standalone_question}")
    if reply.steps:
        with st.expander(f"🛠️ Agent steps ({len(reply.steps)})"):
            for n, step in enumerate(reply.steps, start=1):
                st.code(f"{n}. {step}", language=None)


# --------------------------------------------------------------------------- #
# Ask tab: answer + page + exact line
# --------------------------------------------------------------------------- #
with ask_tab:
    question = st.text_input(
        "Enter a question",
        placeholder="For example: What is the waiting period for pre-existing conditions?",
        key="question",
    )

    if question:
        cache_key = (file_signature, llm_model, top_k, question)
        cache = st.session_state.setdefault("answer_cache", {})
        if cache_key not in cache:
            with st.spinner("Searching the document and preparing a cited answer..."):
                try:
                    cache[cache_key] = answer_question(index, llm, question, top_k=top_k)
                except Exception as exc:
                    st.error(f"Question answering failed: {exc}")
        result = cache.get(cache_key)

        if result is not None:
            st.markdown("### Answer")
            st.write(result.answer)

            st.markdown("### 🔎 Sources (audit trail)")
            if result.citations:
                for n, cite in enumerate(result.citations, start=1):
                    badge = "✅ verified in document" if cite.verified else "⚠️ not found verbatim on this page"
                    st.markdown(f"**[{n}] Page {cite.page}** · {badge}")
                    st.markdown(f"> {cite.quote}")
                if any(not c.verified for c in result.citations):
                    st.warning(
                        "At least one cited line could not be matched word-for-word to the retrieved "
                        "text. Check the passages below before relying on this answer."
                    )
            elif not result.structured:
                st.warning(
                    "The model did not return structured citations for this question. "
                    "The retrieved passages below are the closest sources."
                )
            else:
                st.info("No supporting source was cited — the document may not contain this answer.")

            with st.expander("📖 All retrieved passages", expanded=not result.citations):
                for n, doc in enumerate(result.retrieved_docs, start=1):
                    st.markdown(f"**Passage {n} · Page {page_label(doc.metadata) or 'Unknown'}**")
                    st.caption(doc.page_content)

# --------------------------------------------------------------------------- #
# Chat tab: multi-turn conversation with memory, optional tool-using agent
# --------------------------------------------------------------------------- #
with chat_tab:
    st.markdown(
        "Ask follow-up questions in a conversation, such as *\"And for dependants?\"*. The app rewrites "
        "each message into a standalone question using the recent chat, so the document search "
        "understands it. Every answer keeps the verified page citations."
    )
    use_agent = st.toggle(
        "Agent mode",
        value=False,
        help="The model decides which tools to use: search the document several times, read a "
        "specific page, or calculate numbers. Slower (several model calls) but handles multi-step "
        "questions such as 'What is the annual premium if the monthly premium is on page 3?'.",
    )
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

    history: list[ChatTurn] = st.session_state.setdefault("chat_history", [])
    for turn in history:
        with st.chat_message(turn.role):
            if turn.reply is not None:
                render_reply(turn.reply)
            else:
                st.write(turn.content)

    user_message = st.chat_input("Ask a question about the document...")
    if user_message:
        with st.chat_message("user"):
            st.write(user_message)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = chat_answer(index, llm, user_message, history, top_k=top_k, use_agent=use_agent)
                except Exception as exc:
                    reply = None
                    st.error(f"Could not answer: {exc}")
            if reply is not None:
                render_reply(reply)
                history.append(ChatTurn("user", user_message))
                history.append(ChatTurn("assistant", reply.answer.answer, reply))

# --------------------------------------------------------------------------- #
# Evaluate tab: Question | Bot Answer | Expected Page | Correct?
# --------------------------------------------------------------------------- #
with eval_tab:
    st.markdown(
        "Write test questions about the uploaded PDF and the page where each answer really is. "
        "A question counts as **correct** when the bot cites a *verified* page that matches the "
        "expected page. Separate several valid pages with `|` (e.g. `4|5`)."
    )

    csv_file = st.file_uploader(
        "Optional: load questions from CSV (columns: question, expected_page)", type=["csv"], key="eval_csv"
    )
    if csv_file is not None and st.session_state.get("eval_csv_name") != csv_file.name:
        try:
            loaded = pd.read_csv(csv_file, dtype=str).rename(columns=str.lower)
            st.session_state.eval_questions = loaded[["question", "expected_page"]].fillna("")
            st.session_state.eval_csv_name = csv_file.name
        except Exception as exc:
            st.error(f"Could not read that CSV: {exc}. It needs 'question' and 'expected_page' columns.")

    if "eval_questions" not in st.session_state:
        st.session_state.eval_questions = pd.DataFrame(
            {"question": [""] * 5, "expected_page": [""] * 5}
        )

    edited = st.data_editor(
        st.session_state.eval_questions,
        num_rows="dynamic",
        use_container_width=True,
        key="eval_editor",
        column_config={
            "question": st.column_config.TextColumn("Question", width="large"),
            "expected_page": st.column_config.TextColumn("Expected page", width="small"),
        },
    )

    if st.button("▶ Run evaluation", type="primary"):
        questions = [
            {"question": str(q).strip(), "expected_page": str(p).strip()}
            for q, p in zip(edited["question"], edited["expected_page"])
            if str(q).strip() and str(q).strip().lower() != "nan"
        ]
        missing = [q["question"] for q in questions if not q["expected_page"] or q["expected_page"].lower() == "nan"]
        if not questions:
            st.warning("Add at least one question first.")
        elif missing:
            st.warning("Every question needs an expected page number.")
        else:
            bar = st.progress(0.0, text="Running evaluation...")
            st.session_state.eval_rows = evaluate(
                index,
                llm,
                questions,
                top_k=top_k,
                progress=lambda done, total: bar.progress(done / total, text=f"Question {done} of {total}"),
            )
            bar.empty()

    rows = st.session_state.get("eval_rows")
    if rows:
        stats = score_rows(rows)
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{stats['accuracy']:.0%}", f"{stats['correct']} of {stats['total']} correct")
        col2.metric("Right page retrieved", f"{stats['retrieval_rate']:.0%}")
        col3.metric("Questions", stats["total"])

        table = pd.DataFrame(rows_to_records(rows))
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download results (CSV)",
            table.to_csv(index=False).encode("utf-8"),
            file_name="docuquery_evaluation.csv",
            mime="text/csv",
        )
        st.caption(
            "Accuracy = the cited, verified page matches the expected page. "
            "'Right page retrieved' only checks that retrieval found that page."
        )

st.caption(
    "Note: Answers are generated by an LLM. Always verify important information against the cited source lines."
)
