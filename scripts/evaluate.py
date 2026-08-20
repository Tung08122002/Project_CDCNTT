"""CLI evaluation for a chosen chunking and retrieval mode."""
import argparse, json, sys
from pathlib import Path
import pandas as pd

# Make `python scripts/evaluate.py` work when launched from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.chunking import make_chunks
from rag.evaluation import evaluate_questions
from rag.pdf_ingest import extract_pdf_pages, validate_corpus
from rag.retrieval import SemanticRetriever

p = argparse.ArgumentParser()
p.add_argument("--strategy", choices=["fixed", "recursive"], default="recursive")
p.add_argument("--size", type=int, default=700); p.add_argument("--overlap", type=int, default=120)
args = p.parse_args()
errors = validate_corpus("data/pdfs")
if errors: raise SystemExit("\n".join(errors))
pages = [x for f in Path("data/pdfs").glob("*.pdf") for x in extract_pdf_pages(f)]
retriever = SemanticRetriever(make_chunks(pages, args.strategy, args.size, args.overlap), sections=pages)
evaluation_path = next(
    (path for path in (Path("data/evaluation_questions.json"), Path("data/evaluation_questions.template.json")) if path.exists()),
    None,
)
if evaluation_path is None:
    raise SystemExit("Không tìm thấy data/evaluation_questions.json hoặc file template.")
questions = json.loads(evaluation_path.read_text(encoding="utf-8"))
df = pd.DataFrame(evaluate_questions(retriever, questions))
print(df.to_string(index=False)); print("\nAVERAGE\n", df.select_dtypes("number").mean().round(3))
