from typing import Any, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from app.domain.models.chunk import Chunk


class VectorRepository:
    def __init__(
        self,
        collection_name: str = "acquired_transcripts",
        persist_directory: str = "data/chroma_db",
        batch_size: int = 5000,
    ) -> None:
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        return self.client.get_or_create_collection(name=self.collection_name)

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        total = len(chunks)

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = chunks[start:end]

            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.text for chunk in batch]
            embeddings = [chunk.embedding for chunk in batch]
            metadatas = [self._chunk_metadata(chunk) for chunk in batch]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            print(f"[ChromaDB] Upserted batch {start}-{end} / {total}")

    def query_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }

        if where:
            query_params["where"] = where

        return self.collection.query(**query_params)
    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _chunk_metadata(chunk: Chunk) -> dict[str, Any]:
        return {
            "document_id": chunk.document_id,
            "file_name": chunk.file_name,
            "chunk_index": chunk.chunk_index,
            "title": chunk.title,
            "post_url": chunk.post_url,
            "blog_title": chunk.blog_title,
            "series_number": chunk.series_number,
            "blog_date": chunk.blog_date,
            "token_count": chunk.token_count,
        }