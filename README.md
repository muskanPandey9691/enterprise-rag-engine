# DocuQuery — RAG-based PDF Compliance Assistant

DocuQuery lets you upload a PDF (a policy, contract, regulation, handbook) and ask questions about it.
Every answer comes with an **audit trail**: the page number and the exact line the answer is based on.
In regulated work such as insurance, an answer you cannot trace to a source cannot be trusted. DocuQuery
makes the source part of the answer.

## What it does

1. **Indexes the PDF page by page.** Each text chunk keeps `page` and `source` metadata.
2. **Retrieves** the most relevant chunks with Gemini embeddings and a Chroma vector store.
3. **Answers with citations.** The model returns the answer plus, for each fact, a page number and the exact
   line it relied on.
4. **Verifies every citation.** The app checks that the quoted line really appears, word for word, on the cited
   page. Each source is marked ✅ verified or ⚠️ not found verbatim, so a made-up quote is caught instead of trusted.
5. **Evaluates itself.** The *Evaluate* tab (and `eval.py`) runs your test questions and reports
   Question | Bot Answer | Expected Page | Correct? plus an accuracy score.

## How accuracy is scored

A question counts as **correct** when the bot cites a *verified* page that matches the page you expected.
The app also reports *right page retrieved*, which only checks that retrieval found the expected page. If that
number is high but accuracy is low, the problem is in answering, not retrieval.

## Run locally

```bash
git clone https://github.com/muskanPandey9691/enterprise-rag-engine.git
cd enterprise-rag-engine
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), paste your
[Gemini API key](https://aistudio.google.com/apikey) in the sidebar, and upload a PDF.

## Run an evaluation

In the app: open the **Evaluate** tab, type questions and the page each answer is on (use `4|5` if two pages are
valid), then click **Run evaluation**. You can also load a CSV and download the results.

From the command line:

```bash
export GOOGLE_API_KEY=your-key          # Windows PowerShell: $env:GOOGLE_API_KEY="your-key"
cp eval/questions_template.csv eval/questions.csv   # then write your own questions
python eval.py your_document.pdf eval/questions.csv --out eval/results.csv
```

The template's example rows are placeholders. Replace them with real questions about your own PDF. Only quote
an accuracy figure that comes from an actual run.

### Results

| PDF | Questions | Correct | Accuracy |
|-----|-----------|---------|----------|
| _fill in after your run_ | | | |

## Tests

```bash
pip install pytest
python -m pytest tests
```

The tests cover citation verification and scoring and need no API key.

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, pick the repository, the branch, and `app.py`.
4. Deploy, then enter your Gemini API key in the app's sidebar.

If the app is already deployed, pushing to the deployed branch redeploys it and the link stays the same.

Do not commit API keys, credentials, private PDFs, or local database files to GitHub.

## Project structure

```text
enterprise-rag-engine/
├── app.py                      # Streamlit UI: Ask tab + Evaluate tab
├── rag_core.py                 # Indexing, retrieval, cited answers, evaluation loop
├── citations.py                # Quote verification and scoring (no API needed)
├── eval.py                     # Command-line evaluation
├── eval/questions_template.csv # Question set template
├── tests/test_citations.py
├── requirements.txt
└── README.md
```

## Limitations

- One PDF at a time. Uploading a different PDF or changing chunk settings rebuilds the index.
- The PDF must contain selectable text. Scanned PDFs need OCR first.
- Verification confirms the quoted line exists on the cited page. It does not prove the line fully supports the
  answer, so a human should still read the cited line for important decisions.
- Page numbers are PDF page positions (page 1 is the first page), not the numbers printed on the page.
- Model names can change. If Gemini access changes, edit the model name in the sidebar or `DEFAULT_LLM_MODEL` in `rag_core.py`.
