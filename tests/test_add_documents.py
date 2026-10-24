import sys
import types

import pytest

# src.indexing imports chromadb and sentence_transformers at module level.
# Stub them so this test runs without the heavy runtime dependencies.
sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))
_st = types.ModuleType("sentence_transformers")
_st.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", _st)

from src.add_documents import add_articles, validate_articles


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def delete(self, where):
        number = where["article_number"]
        self.rows = {
            key: value
            for key, value in self.rows.items()
            if value["meta"]["article_number"] != number
        }

    def upsert(self, ids, documents, embeddings, metadatas):
        for chunk_id, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            self.rows[chunk_id] = {"doc": doc, "emb": emb, "meta": meta}

    def count(self):
        return len(self.rows)


def fake_embed(texts):
    return [[float(len(text))] for text in texts]


NEW_ARTICLE = {
    "article_number": 5000,
    "ar_text": "نص تجريبي لمادة جديدة.",
    "text_en": "Sample text for a new article.",
    "citation": "Egyptian Civil Code, Article 5000",
}


def test_new_article_is_added_with_citation():
    collection = FakeCollection()
    written = add_articles([NEW_ARTICLE], collection, fake_embed)
    assert written == 1
    assert collection.count() == 1
    row = collection.rows["article-5000-chunk-0"]
    assert row["meta"]["article_number"] == 5000
    assert row["meta"]["citation"] == "Egyptian Civil Code, Article 5000"


def test_running_twice_does_not_duplicate():
    collection = FakeCollection()
    add_articles([NEW_ARTICLE], collection, fake_embed)
    add_articles([NEW_ARTICLE], collection, fake_embed)
    assert collection.count() == 1


def test_shorter_update_removes_stale_chunks():
    collection = FakeCollection()
    long_article = dict(NEW_ARTICLE, ar_text=("فقرة طويلة جدا.\n" * 40))
    add_articles([long_article], collection, fake_embed, max_chars=100, overlap_chars=10)
    assert collection.count() > 1
    add_articles([NEW_ARTICLE], collection, fake_embed)
    assert collection.count() == 1


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        validate_articles([])
    with pytest.raises(TypeError):
        validate_articles([{"article_number": "5", "ar_text": "x"}])
    with pytest.raises(ValueError):
        validate_articles([{"article_number": 5, "ar_text": "  "}])
    with pytest.raises(ValueError):
        validate_articles([NEW_ARTICLE, NEW_ARTICLE])
