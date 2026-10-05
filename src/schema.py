from pydantic import BaseModel, Field


class Article(BaseModel):
    article_number: int
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    ar_text: str
    text_en: str | None = None
    is_repealed: bool = False
    source_page: int | None = None
    citation: str

    def display_citation(self) -> str:
        return self.citation or f"Egyptian Civil Code, Article {self.article_number}"