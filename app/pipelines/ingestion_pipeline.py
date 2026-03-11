import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from app.domain.models.document import Document
from app.domain.models.ingestion_result import IngestionResult
from app.domain.models.qa_record import QARecord


class IngestionPipeline:
    def __init__(
        self,
        data_dir: str = "data",
        transcripts_dir_name: str = "acquired-individual-transcripts",
        processed_dir_name: str = "processed",
        metadata_file_name: str = "acquired_metadata.csv",
        qa_file_name: str = "acquired-qa-evaluation.csv",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transcripts_dir = self.data_dir / transcripts_dir_name
        self.processed_dir = self.data_dir / processed_dir_name
        self.metadata_file = self.data_dir / metadata_file_name
        self.qa_file = self.data_dir / qa_file_name

    def run(self) -> IngestionResult:
        self._ensure_processed_dir()

        metadata_rows = self.load_metadata()
        qa_rows = self.load_qa_rows()

        transcript_index = self.index_transcript_files()

        documents, missing_transcripts, metadata_without_transcript_file = self.build_documents(
            metadata_rows=metadata_rows,
            transcript_index=transcript_index,
        )

        qa_records, qa_without_matching_document = self.build_qa_records(
            qa_rows=qa_rows,
            valid_document_ids={doc.document_id for doc in documents},
        )

        self.save_documents(documents)
        self.save_qa_records(qa_records)

        result = IngestionResult(
            metadata_rows=len(metadata_rows),
            transcript_files_found=len(transcript_index),
            documents_created=len(documents),
            qa_records_loaded=len(qa_records),
            missing_transcripts=missing_transcripts,
            metadata_without_transcript_file=metadata_without_transcript_file,
            qa_without_matching_document=qa_without_matching_document,
        )

        self.save_report(result)
        return result

    def _ensure_processed_dir(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_metadata(self) -> List[dict]:
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        with self.metadata_file.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            return [self._normalize_row(row) for row in reader]

    def load_qa_rows(self) -> List[dict]:
        if not self.qa_file.exists():
            raise FileNotFoundError(f"QA file not found: {self.qa_file}")

        with self.qa_file.open("r", encoding="latin-1", newline="") as file:
            reader = csv.DictReader(file)
            return [self._normalize_row(row) for row in reader]

    def index_transcript_files(self) -> Dict[str, Path]:
        if not self.transcripts_dir.exists():
            raise FileNotFoundError(f"Transcripts directory not found: {self.transcripts_dir}")

        transcript_index: Dict[str, Path] = {}

        for path in self.transcripts_dir.rglob("*.txt"):
            normalized_name = path.stem.strip().lower()
            transcript_index[normalized_name] = path

        return transcript_index

    def build_documents(
        self,
        metadata_rows: List[dict],
        transcript_index: Dict[str, Path],
    ) -> tuple[List[Document], List[str], List[str]]:
        documents: List[Document] = []
        missing_transcripts: List[str] = []
        metadata_without_transcript_file: List[str] = []

        for row in metadata_rows:
            raw_file_name = row.get("file_name")
            normalized_file_name = self._normalize_file_name(raw_file_name)

            if not normalized_file_name:
                metadata_without_transcript_file.append(str(raw_file_name))
                continue

            transcript_path = transcript_index.get(normalized_file_name)
            if transcript_path is None:
                missing_transcripts.append(normalized_file_name)
                continue

            content = self.load_transcript_text(transcript_path)
            if not content:
                missing_transcripts.append(normalized_file_name)
                continue

            document = Document(
                document_id=normalized_file_name,
                file_name=normalized_file_name,
                title=row.get("post_title"),
                post_url=row.get("post_url"),
                series_number=row.get("series_number"),
                blog_date=row.get("blog_date"),
                blog_title=row.get("blog_title"),
                content=content,
            )
            documents.append(document)

        return documents, missing_transcripts, metadata_without_transcript_file

    def build_qa_records(
        self,
        qa_rows: List[dict],
        valid_document_ids: set[str],
    ) -> tuple[List[QARecord], List[str]]:
        qa_records: List[QARecord] = []
        qa_without_matching_document: List[str] = []

        for index, row in enumerate(qa_rows, start=1):
            normalized_file_name = self._normalize_file_name(row.get("file_name"))

            question_id = f"qa_{index:05d}"

            qa_record = QARecord(
                question_id=question_id,
                question=row.get("question") or "",
                human_answer=row.get("human_answer"),
                ai_answer_without_the_transcript=row.get("ai_answer_without_the_transcript"),
                ai_answer_without_transcript_correctness=row.get("ai_answer_without_transcript_correctness"),
                ai_answer_with_the_transcript=row.get("ai_answer_with_the_transcript"),
                ai_answer_with_the_transcript_correctness=row.get("ai_answer_with_the_transcript_correctness"),
                quality_rating_for_answer_with_transcript=row.get("quality_rating_for_answer_with_transcript"),
                post_url=row.get("post_url"),
                file_name=normalized_file_name,
            )
            qa_records.append(qa_record)

            if normalized_file_name and normalized_file_name not in valid_document_ids:
                qa_without_matching_document.append(normalized_file_name)

        return qa_records, sorted(set(qa_without_matching_document))

    def load_transcript_text(self, transcript_path: Path) -> str:
        text = transcript_path.read_text(encoding="utf-8", errors="ignore")
        return self._normalize_text(text)

    def save_documents(self, documents: List[Document]) -> None:
        output_path = self.processed_dir / "documents.jsonl"
        with output_path.open("w", encoding="utf-8") as file:
            for document in documents:
                file.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")

    def save_qa_records(self, qa_records: List[QARecord]) -> None:
        output_path = self.processed_dir / "qa_records.jsonl"
        with output_path.open("w", encoding="utf-8") as file:
            for qa_record in qa_records:
                file.write(json.dumps(qa_record.to_dict(), ensure_ascii=False) + "\n")

    def save_report(self, result: IngestionResult) -> None:
        output_path = self.processed_dir / "ingestion_report.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result.to_dict(), file, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_row(row: dict) -> dict:
        normalized = {}
        for key, value in row.items():
            clean_key = key.strip() if isinstance(key, str) else key
            normalized[clean_key] = IngestionPipeline._normalize_optional_string(value)
        return normalized

    @staticmethod
    def _normalize_optional_string(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        cleaned = value.strip()
        return cleaned if cleaned else None

    @staticmethod
    def _normalize_file_name(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        normalized = value.strip().lower()
        if normalized.endswith(".txt"):
            normalized = normalized[:-4]

        return normalized or None

    @staticmethod
    def _normalize_text(text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        cleaned = " ".join(line for line in lines if line)
        cleaned = " ".join(cleaned.split())
        return cleaned