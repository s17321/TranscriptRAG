from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class IngestionResult:
    metadata_rows: int
    transcript_files_found: int
    documents_created: int
    qa_records_loaded: int
    missing_transcripts: List[str] = field(default_factory=list)
    metadata_without_transcript_file: List[str] = field(default_factory=list)
    qa_without_matching_document: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)