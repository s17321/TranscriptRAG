from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class QARecord:
    question_id: str
    question: str
    human_answer: Optional[str]
    ai_answer_without_the_transcript: Optional[str]
    ai_answer_without_transcript_correctness: Optional[str]
    ai_answer_with_the_transcript: Optional[str]
    ai_answer_with_the_transcript_correctness: Optional[str]
    quality_rating_for_answer_with_transcript: Optional[str]
    post_url: Optional[str]
    file_name: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)