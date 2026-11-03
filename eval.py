"""Run the DocuQuery evaluation from the command line.

    export GOOGLE_API_KEY=your-key
    python eval.py my_policy.pdf eval/questions.csv --out eval/results.csv

questions.csv needs two columns: question, expected_page
"""

import argparse
import csv
import os
import sys

from citations import rows_to_records, score_rows
from rag_core import DEFAULT_LLM_MODEL, build_index, evaluate, make_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DocuQuery on a PDF and a question set.")
    parser.add_argument("pdf", help="Path to the PDF to index")
    parser.add_argument("questions", help="CSV with columns: question, expected_page")
    parser.add_argument("--out", default="eval/results.csv", help="Where to write the results CSV")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="Gemini chat model name")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set the GOOGLE_API_KEY environment variable first.", file=sys.stderr)
        return 1

    with open(args.questions, newline="", encoding="utf-8-sig") as handle:
        questions = [
            {k.strip().lower(): v for k, v in row.items()} for row in csv.DictReader(handle)
        ]
    if not questions:
        print("No questions found in the CSV.", file=sys.stderr)
        return 1

    embeddings, llm = make_models(api_key, args.model)
    with open(args.pdf, "rb") as handle:
        index = build_index(
            handle.read(),
            os.path.basename(args.pdf),
            embeddings,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    print(f"Indexed {index.name}: {index.page_count} pages, {index.chunk_count} chunks")

    rows = evaluate(
        index,
        llm,
        questions,
        top_k=args.top_k,
        progress=lambda done, total: print(f"  question {done}/{total}", end="\r"),
    )
    print()

    records = rows_to_records(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    stats = score_rows(rows)
    print(f"Accuracy: {stats['correct']}/{stats['total']} = {stats['accuracy']:.0%}")
    print(f"Right page retrieved: {stats['retrieval_hits']}/{stats['total']} = {stats['retrieval_rate']:.0%}")
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
