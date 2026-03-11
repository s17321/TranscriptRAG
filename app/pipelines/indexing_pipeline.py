import json
from pathlib import Path
from typing import List

from app.domain.models.chunk import Chunk
from app.domain.models.document import Document
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService


class IndexingPipeline:
    def __init__(
        self,
        data_dir: str = "data",
        processed_dir_name: str = "processed",
        documents_file_name: str = "documents.jsonl",
        chunks_file_name: str = "chunks.jsonl",
        chunk_size=500,
        chunk_overlap=100,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / processed_dir_name
        self.documents_file = self.processed_dir / documents_file_name
        self.chunks_file = self.processed_dir / chunks_file_name

        self.chunking_service = ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_service = EmbeddingService(
            model_name=embedding_model_name
        )

    def run(self) -> dict:
        documents = self.load_documents()
        chunks = self.build_chunks(documents)
        self.save_chunks(chunks)

        return {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "output_file": str(self.chunks_file),
        }

    def load_documents(self) -> List[Document]:
        if not self.documents_file.exists():
            raise FileNotFoundError(f"Documents file not found: {self.documents_file}")

        documents: List[Document] = []

        with self.documents_file.open("r", encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                document = Document(**row)
                documents.append(document)

        return documents

    def build_chunks(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []

        for document in documents:
            split_chunks = self.chunking_service.split_text(document.content)
            if not split_chunks:
                continue

            embeddings = self.embedding_service.generate_embeddings(split_chunks)

            for index, (chunk_text, embedding) in enumerate(zip(split_chunks, embeddings)):
                chunk = Chunk(
                    chunk_id=f"{document.document_id}_chunk_{index:05d}",
                    document_id=document.document_id,
                    file_name=document.file_name,
                    chunk_index=index,
                    text=chunk_text,
                    title=document.title,
                    post_url=document.post_url,
                    blog_title=document.blog_title,
                    series_number=document.series_number,
                    blog_date=document.blog_date,
                    token_count=len(chunk_text.split()),
                    embedding=embedding,
                )
                chunks.append(chunk)

        return chunks

    def save_chunks(self, chunks: List[Chunk]) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        with self.chunks_file.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
