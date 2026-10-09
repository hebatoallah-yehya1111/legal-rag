import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
    VECTOR_COLLECTION_NAME,
)


def load_chunks(input_path: str):
    """Load chunks from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_embedding_text(chunk: dict) -> str:
    """
    Build bilingual text representation for embedding.

    Both Arabic and English are included so that:
    - Arabic questions can retrieve the relevant chunk.
    - English questions can retrieve the same chunk.
    """
    ar_text = (chunk.get("ar_text") or "").strip()
    en_text = (chunk.get("text_en") or "").strip()

    parts = []

    if ar_text:
        parts.append(f"Arabic:\n{ar_text}")

    if en_text:
        parts.append(f"English:\n{en_text}")

    return "\n\n".join(parts)


def build_metadata(chunk: dict) -> dict:
    """Build metadata stored alongside each vector."""
    return {
        "article_number": int(chunk.get("article_number", -1)),
        "citation": chunk.get("citation", ""),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "chunk_count": int(chunk.get("chunk_count", 1)),
        "is_repealed": bool(chunk.get("is_repealed", False)),
        "source_page": int(chunk.get("source_page", -1)),
        "section": chunk.get("section") or "",
        "topic": chunk.get("topic") or "",
    }


def build_index(
    input_path: str,
    output_dir: str,
    batch_size: int = 32,
):
    print(f"Loading chunks from: {input_path}")

    chunks = load_chunks(input_path)

    if not chunks:
        raise ValueError("No chunks found in input file.")

    print(f"Loaded {len(chunks)} chunks.")

    # ---------------------------------------------------------
    # 1. Build bilingual text for every chunk
    # ---------------------------------------------------------
    documents = []

    for chunk in chunks:
        embedding_text = build_embedding_text(chunk)

        if not embedding_text:
            raise ValueError(
                f"Chunk {chunk.get('chunk_id')} has no Arabic or English text."
            )

        documents.append(embedding_text)

    print("Built bilingual embedding texts.")

    # ---------------------------------------------------------
    # 2. Load multilingual embedding model
    # ---------------------------------------------------------
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    # ---------------------------------------------------------
    # 3. Generate embeddings
    # ---------------------------------------------------------
    print("Generating bilingual embeddings...")

    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"Generated {len(embeddings)} embeddings.")

    # ---------------------------------------------------------
    # 4. Create persistent ChromaDB
    # ---------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating ChromaDB at: {output_path}")

    client = chromadb.PersistentClient(path=str(output_path))

    # Recreate collection to make indexing reproducible
    try:
        client.delete_collection(VECTOR_COLLECTION_NAME)
        print(f"Deleted existing collection: {VECTOR_COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=VECTOR_COLLECTION_NAME,
        metadata={
            "description": "Bilingual Arabic-English Egyptian Civil Code RAG index",
            "embedding_model": EMBEDDING_MODEL,
            "language_support": "ar,en",
        },
    )

    # ---------------------------------------------------------
    # 5. Prepare Chroma records
    # ---------------------------------------------------------
    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            chunk_id = f"chunk-{i}"

        ids.append(chunk_id)
        metadatas.append(build_metadata(chunk))

    # ---------------------------------------------------------
    # 6. Add vectors to ChromaDB in batches
    # ---------------------------------------------------------
    print("Adding vectors to ChromaDB...")

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

        print(f"Indexed {end}/{len(chunks)} chunks.")

    # ---------------------------------------------------------
    # 7. Verify
    # ---------------------------------------------------------
    count = collection.count()

    print("\nIndexing completed successfully.")
    print(f"Collection: {VECTOR_COLLECTION_NAME}")
    print(f"Documents indexed: {count}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Languages: Arabic + English")


def main():
    parser = argparse.ArgumentParser(
        description="Build bilingual Arabic-English ChromaDB index."
    )

    parser.add_argument(
        "--input",
        default="data/processed/chunks.json",
        help="Path to chunks JSON file.",
    )

    parser.add_argument(
        "--out",
        default=VECTOR_DB_DIR,
        help="Output directory for ChromaDB.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding/indexing batch size.",
    )

    args = parser.parse_args()

    build_index(
        input_path=args.input,
        output_dir=args.out,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()