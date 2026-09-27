"""Data schemas for the Egyptian Civil Code corpus and the API."""

from pydantic import BaseModel, Field


class Article(BaseModel):
    """One record per article of the Egyptian Civil Code."""

    article_number: int
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    text_ar: str
    text_en: str | None = None
    is_repealed: bool = False
    source_page: int | None = None
    citation: str

    def display_citation(self) -> str:
        return self.citation or f"Egyptian Civil Code, Article {self.article_number}"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
