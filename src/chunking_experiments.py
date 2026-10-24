"""Compare chunking settings and log every run to MLflow.

Each run rebuilds the chunks with one (max_chars, overlap_chars, embedding_model)
setting, indexes them in memory (the DVC-tracked vector store is not touched),
and measures retrieval quality on the gold questions (Hit@k and MRR).
No LLM call and no API quota are needed.

Embeddings are cached by text, so settings that leave most chunks unchanged
are much faster after the first run.

Usage:
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000
    python -m src.chunking_experiments
    python -m src.chunking_experiments --register-best
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.chunking import make_chunks
from src.evaluate_retrieval import evaluate, summarize

ROOT = Path(__file__).resolve().parents[1]
ARTICLES = ROOT / "data" / "processed" / "articles.json"
GOLD = ROOT / "data" / "evaluation" / "gold_articles.json"

# (max_chars, overlap_chars). Most articles are short, so only the long ones change.
DEFAULT_CONFIGS = [(4000, 300), (2000, 200), (1000, 150), (600, 100), (400, 50)]


def run_config(model, model_name, articles, gold_rows, max_chars, overlap_chars, cache):
    import chromadb

    from src.indexing import build_embedding_text, build_metadata

    chunks = make_chunks(articles, max_chars=max_chars, overlap_chars=overlap_chars)
    documents = [build_embedding_text(chunk) for chunk in chunks]

    missing = [doc for doc in set(documents) if (model_name, doc) not in cache]
    if missing:
        vectors = model.encode(missing, batch_size=32, show_progress_bar=True,
                               normalize_embeddings=True)
        for doc, vector in zip(missing, vectors):
            cache[(model_name, doc)] = vector.tolist()

    client = chromadb.EphemeralClient()
    name = f"exp_{max_chars}_{overlap_chars}_{abs(hash(model_name)) % 10000}"
    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    batch = 256
    for start in range(0, len(chunks), batch):
        end = start + batch
        collection.add(
            ids=[c["chunk_id"] for c in chunks[start:end]],
            documents=documents[start:end],
            embeddings=[cache[(model_name, d)] for d in documents[start:end]],
            metadatas=[build_metadata(c) for c in chunks[start:end]],
        )

    def search(question: str, k: int):
        vector = model.encode(question, normalize_embeddings=True).tolist()
        result = collection.query(query_embeddings=[vector], n_results=k,
                                  include=["metadatas"])
        return [{"article_number": m.get("article_number")} for m in result["metadatas"][0]]

    results = evaluate(gold_rows, search, k_max=10)
    return len(chunks), summarize([r["rank"] for r in results]), len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", action="append", default=None,
                        help="Repeat to compare several models. Default: the model in config.")
    parser.add_argument("--register-best", action="store_true",
                        help="Register the best run (by MRR) in the MLflow Model Registry.")
    parser.add_argument("--experiment", default="legal-rag-chunking")
    args = parser.parse_args()

    import mlflow
    import mlflow.pyfunc
    from sentence_transformers import SentenceTransformer

    from src.config import EMBEDDING_MODEL, MLFLOW_TRACKING_URI

    models = args.embedding_model or [EMBEDDING_MODEL]
    articles = json.loads(ARTICLES.read_text(encoding="utf-8"))
    gold_rows = json.loads(GOLD.read_text(encoding="utf-8"))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment)

    cache: dict = {}
    best = None

    for model_name in models:
        model = SentenceTransformer(model_name)
        for max_chars, overlap_chars in DEFAULT_CONFIGS:
            with mlflow.start_run(run_name=f"chunk{max_chars}-ov{overlap_chars}-{model_name.split('/')[-1]}") as run:
                chunk_count, summary, n_questions = run_config(
                    model, model_name, articles, gold_rows, max_chars, overlap_chars, cache
                )
                params = {
                    "chunk_size": max_chars,
                    "overlap": overlap_chars,
                    "embedding_model": model_name,
                    "chunk_strategy": "by_article_split_by_paragraph",
                    "n_questions": n_questions,
                }
                mlflow.log_params(params)
                mlflow.log_metrics({**summary, "chunk_count": chunk_count})
                print(f"{params} -> {summary}")
                if best is None or summary["mrr"] > best["mrr"]:
                    best = {"run_id": run.info.run_id, "mrr": summary["mrr"], "params": params}

    print(f"\nBest run: {best}")

    if args.register_best and best:
        class ChunkingConfig(mlflow.pyfunc.PythonModel):
            """Carries the chosen chunking configuration so it can live in the Registry."""

            def __init__(self, config):
                self.config = config

            def predict(self, context, model_input, params=None):
                return self.config

        with mlflow.start_run(run_id=best["run_id"]):
            info = mlflow.pyfunc.log_model(
                artifact_path="chunking_config",
                python_model=ChunkingConfig(best["params"]),
            )
        version = mlflow.register_model(info.model_uri, "legal-rag-chunking-config")
        print(f"Registered version {version.version} of legal-rag-chunking-config")


if __name__ == "__main__":
    main()
