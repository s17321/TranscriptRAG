import json
from pathlib import Path
from typing import List

from app.domain.models.chunk import Chunk
from app.repositories.vector_repository import VectorRepository


def load_chunks(chunks_file: str = "data/processed/chunks.jsonl") -> List[Chunk]:
    path = Path(chunks_file)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            chunks.append(Chunk(**row))

    return chunks


def main() -> None:
    chunks = load_chunks()
    repository = VectorRepository()

    repository.upsert_chunks(chunks)

    print("Vector indexing finished successfully.")
    print(f"Chunks loaded from file: {len(chunks)}")
    print(f"Chunks stored in ChromaDB: {repository.count()}")


if __name__ == "__main__":
    main()