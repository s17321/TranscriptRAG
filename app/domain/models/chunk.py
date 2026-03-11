from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    file_name: str
    chunk_index: int
    text: str
    title: Optional[str] = None
    post_url: Optional[str] = None
    blog_title: Optional[str] = None
    series_number: Optional[str] = None
    blog_date: Optional[str] = None
    token_count: Optional[int] = None
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)