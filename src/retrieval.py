import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_DIR,
)


class Retriever:
    def __init__(
        self,
        vector_db_dir: str = VECTOR_DB_DIR,
        collection_name: str = VECTOR_COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(
            path=vector_db_dir
        )

        self.collection = self.client.get_collection(
            name=collection_name
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        """
        Retrieve the most relevant legal chunks for a query.
        Supports Arabic and English queries.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query = query.strip()

        # Convert the user's question into the same
        # multilingual embedding space used during indexing.
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved_chunks = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for i in range(len(documents)):
            metadata = metadatas[i] or {}

            retrieved_chunks.append(
                {
                    "chunk_id": ids[i],
                    "document": documents[i],
                    "distance": distances[i],
                    "article_number": metadata.get("article_number"),
                    "citation": metadata.get("citation"),
                    "source_page": metadata.get("source_page"),
                    "section": metadata.get("section"),
                    "topic": metadata.get("topic"),
                    "is_repealed": metadata.get("is_repealed"),
                }
            )

        return retrieved_chunks


def print_results(query: str, results: list):
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Article: {result['article_number']}")
        print(f"Citation: {result['citation']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Source page: {result['source_page']}")
        print(f"Section: {result['section']}")
        print(f"Topic: {result['topic']}")
        print("\nDocument:")
        print(result["document"])


def main():
    parser = argparse.ArgumentParser(
        description="Test bilingual legal document retrieval."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Arabic or English legal question.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve.",
    )

    args = parser.parse_args()

    retriever = Retriever()

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
    )

    print_results(
        query=args.query,
        results=results,
    )


if __name__ == "__main__":
    main()