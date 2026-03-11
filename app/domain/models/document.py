from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Document:
    document_id: str
    file_name: str
    title: Optional[str]
    post_url: Optional[str]
    series_number: Optional[str]
    blog_date: Optional[str]
    blog_title: Optional[str]
    content: str
    source_type: str = "podcast_transcript"

    def to_dict(self) -> dict:
        return asdict(self)