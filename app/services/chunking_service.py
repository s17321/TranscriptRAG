from typing import List


class ChunkingService:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        normalized_text = " ".join(text.split())
        if not normalized_text:
            return []

        words = normalized_text.split()
        chunks: List[str] = []

        step = self.chunk_size - self.chunk_overlap

        for start_idx in range(0, len(words), step):
            chunk_words = words[start_idx:start_idx + self.chunk_size]
            if not chunk_words:
                continue

            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunks.append(chunk_text)

            if start_idx + self.chunk_size >= len(words):
                break

        return chunks