"""
Dataset preparation: turn your raw materials into the JSONL the harness expects.

Inputs (yours):
  * a folder of PDFs           -> the knowledge base
  * a CSV/XLSX of Q&A pairs    -> one row per question

Outputs (what the harness reads):
  * corpus.jsonl    {"doc_id", "text", "source_uri"}
  * evalset.jsonl   {"item_id", "query", "item_type", "gold_answer", "gold_passage_ids"}

The clever bit: because your gold answers are EXACT short strings/numbers, this
tool auto-fills `gold_passage_ids` by finding which chunk actually contains each
answer. That recovers the retrieval metrics (hit-rate@k, MRR, ...) with no manual
passage labeling. It uses the harness's OWN chunker, so the chunk ids it writes
match exactly what `main.py ingest` will create in Qdrant.

Usage:
  python prepare_dataset.py \
      --pdf-dir  ./my_pdfs \
      --qa-file  ./my_questions.xlsx \
      --out-dir  ./data/regulated_qa \
      --question-col Question --answer-col Answer \
      --chunk-size 200 --overlap 40

Requires: pdfplumber, pandas, openpyxl (for .xlsx). Install:
  pip install pdfplumber openpyxl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# Reuse the harness chunker so chunk ids line up with ingestion exactly.
from harness.rag.ingest import chunk_text


# --------------------------------------------------------------------------- #
#  1. PDFs -> corpus.jsonl                                                     #
# --------------------------------------------------------------------------- #
def extract_pdf_text(pdf_path: Path, granularity: str) -> list[tuple[str, str]]:
    """
    Return [(doc_id, text), ...] for one PDF.

    granularity="page"     -> one record per page  (recommended: better retrieval
                              AND tighter answer-to-passage localisation)
    granularity="document" -> one record for the whole PDF (simplest)

    Uses pdfplumber (layout-aware). If a page yields no text, the PDF is likely
    scanned — this prints a warning so you know to OCR it separately rather than
    silently indexing empty documents.
    """
    import pdfplumber

    stem = pdf_path.stem
    records: list[tuple[str, str]] = []
    empty_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        if granularity == "document":
            parts = []
            for page in pdf.pages:
                t = page.extract_text() or ""
                if not t.strip():
                    empty_pages += 1
                parts.append(t)
            full = "\n".join(parts).strip()
            if full:
                records.append((stem, full))
        else:  # page-level
            for i, page in enumerate(pdf.pages):
                t = (page.extract_text() or "").strip()
                if not t:
                    empty_pages += 1
                    continue
                records.append((f"{stem}_p{i+1}", t))

    if empty_pages:
        print(f"  ! {pdf_path.name}: {empty_pages} page(s) had no extractable "
              f"text (scanned? -> OCR needed for those).")
    return records


def build_corpus(pdf_dir: Path, out_path: Path, granularity: str) -> list[dict]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")
    print(f"[corpus] extracting {len(pdfs)} PDF(s), granularity={granularity}")

    corpus: list[dict] = []
    seen_ids: set[str] = set()
    for pdf in pdfs:
        for doc_id, text in extract_pdf_text(pdf, granularity):
            # guarantee unique doc_ids even if two PDFs share a stem
            uid = doc_id
            n = 1
            while uid in seen_ids:
                uid = f"{doc_id}_{n}"
                n += 1
            seen_ids.add(uid)
            corpus.append({
                "doc_id": uid,
                "text": text,
                "source_uri": f"{pdf.name}",
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in corpus:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[corpus] wrote {len(corpus)} documents -> {out_path}")
    return corpus


# --------------------------------------------------------------------------- #
#  2. Auto-label gold_passage_ids via exact-answer matching                    #
# --------------------------------------------------------------------------- #
def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def build_chunk_index(corpus: list[dict], chunk_size: int,
                      overlap: int) -> list[tuple[str, str]]:
    """
    Recreate the EXACT chunks the harness will index, as [(chunk_id, chunk_text)].
    chunk_id format is doc_id#index, matching harness.rag.ingest.
    """
    index: list[tuple[str, str]] = []
    for doc in corpus:
        for i, ch in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            index.append((f"{doc['doc_id']}#{i}", ch))
    return index


def find_gold_passages(answer: str, chunk_index: list[tuple[str, str]],
                       max_matches: int = 3) -> list[str]:
    """
    Return chunk_ids whose text contains the exact (normalised) answer string.
    For numeric answers we also try a digits-only match so "5,000" finds "5000".
    Returns up to max_matches ids (a unique match is ideal; several means the
    answer appears in multiple places and the label is weaker but still useful).
    """
    if answer is None or str(answer).strip() == "":
        return []
    norm_ans = _normalise(answer)
    hits = [cid for cid, ctext in chunk_index if norm_ans in _normalise(ctext)]

    # numeric fallback: compare digit sequences ("5,000" ~ "5000")
    if not hits:
        digits = re.sub(r"[^\d]", "", str(answer))
        if digits:
            hits = [cid for cid, ctext in chunk_index
                    if digits in re.sub(r"[^\d]", "", ctext)]
    return hits[:max_matches]


# --------------------------------------------------------------------------- #
#  3. Q&A spreadsheet -> evalset.jsonl                                         #
# --------------------------------------------------------------------------- #
def build_evalset(qa_file: Path, corpus: list[dict], out_path: Path,
                  question_col: str, answer_col: str,
                  chunk_size: int, overlap: int) -> None:
    # read CSV or Excel transparently
    if qa_file.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(qa_file)
    else:
        df = pd.read_csv(qa_file)

    for col in (question_col, answer_col):
        if col not in df.columns:
            raise SystemExit(
                f"Column '{col}' not in {qa_file.name}. "
                f"Found columns: {list(df.columns)}. "
                f"Pass the right --question-col / --answer-col."
            )

    chunk_index = build_chunk_index(corpus, chunk_size, overlap)
    print(f"[evalset] {len(df)} Q&A rows; {len(chunk_index)} chunks to match against")

    labeled = unlabeled = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            q = str(row[question_col]).strip()
            a = row[answer_col]
            if not q or pd.isna(a):
                continue
            gold_ids = find_gold_passages(str(a), chunk_index)
            if gold_ids:
                labeled += 1
            else:
                unlabeled += 1
            rec = {
                "item_id": f"q{i+1}",
                "query": q,
                "item_type": "answerable",
                "gold_answer": str(a).strip(),
                "gold_passage_ids": gold_ids,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[evalset] wrote {labeled + unlabeled} items -> {out_path}")
    print(f"[evalset]   auto-labeled with gold passages: {labeled}")
    print(f"[evalset]   no passage match (answerable, but no retrieval metrics): "
          f"{unlabeled}")
    if unlabeled:
        print("[evalset]   (these still score accuracy/faithfulness/cost/latency; "
              "they just won't contribute to hit-rate@k / MRR / recall.)")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Prepare corpus + evalset JSONL "
                                            "from PDFs and a Q&A spreadsheet.")
    p.add_argument("--pdf-dir", required=True, type=Path,
                   help="folder containing your source PDFs")
    p.add_argument("--qa-file", required=True, type=Path,
                   help="CSV or XLSX with one Q&A per row")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="where to write corpus.jsonl + evalset.jsonl "
                        "(e.g. data/regulated_qa)")
    p.add_argument("--question-col", default="Question")
    p.add_argument("--answer-col", default="Answer")
    p.add_argument("--granularity", choices=["page", "document"], default="page",
                   help="one corpus record per PDF page (default) or per document")
    # chunk params MUST match what you pass to `main.py ingest` so the auto-labeled
    # gold_passage_ids line up with the chunks actually indexed.
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--overlap", type=int, default=40)
    args = p.parse_args()

    corpus = build_corpus(args.pdf_dir, args.out_dir / "corpus.jsonl",
                          args.granularity)
    build_evalset(args.qa_file, corpus, args.out_dir / "evalset.jsonl",
                  args.question_col, args.answer_col,
                  args.chunk_size, args.overlap)

    print("\nDone. Next:")
    print(f"  1) Set accuracy_scorer to 'exact' or 'numeric' in the profile config")
    print(f"  2) python main.py ingest --profile <your_profile>.yaml "
          f"--chunk-size {args.chunk_size} --overlap {args.overlap}")
    print(f"  3) python main.py run    --profile <your_profile>.yaml")


if __name__ == "__main__":
    main()
