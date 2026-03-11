from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatAskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks returned from retrieval")
    file_name: Optional[str] = Field(
        default=None,
        description="Optional metadata filter for a specific document, e.g. 'airbnb'",
    )


class SourceMetadata(BaseModel):
    document_id: Optional[str] = None
    file_name: Optional[str] = None
    chunk_index: Optional[int] = None
    title: Optional[str] = None
    post_url: Optional[str] = None
    blog_title: Optional[str] = None
    series_number: Optional[str] = None
    blog_date: Optional[str] = None
    token_count: Optional[int] = None


class SourceItem(BaseModel):
    chunk_id: str
    text: str
    metadata: SourceMetadata
    distance: Optional[float] = None
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    final_score: Optional[float] = None


class ChatAskResponse(BaseModel):
    question: str
    answer: str
    model: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    file_name_filter: Optional[str] = None
    sources: list[SourceItem]