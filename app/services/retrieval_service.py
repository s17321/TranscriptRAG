import re
from typing import Any, List, Optional

from app.repositories.vector_repository import VectorRepository
from app.services.embedding_service import EmbeddingService


class RetrievalService:
    def __init__(
        self,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        candidate_k: int = 20,
    ) -> List[dict[str, Any]]:
        query_embedding = self.embedding_service.generate_embedding(question)

        normalized_filters = None
        if filters:
            normalized_filters = {
                key: value.strip().lower() if isinstance(value, str) else value
                for key, value in filters.items()
            }

        results = self.vector_repository.query_similar_chunks(
            query_embedding=query_embedding,
            top_k=candidate_k,
            where=normalized_filters,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_chunks = []

        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            semantic_score = self._semantic_score_from_distance(distance)
            keyword_score = self._keyword_score(question, document)
            final_score = (0.7 * semantic_score) + (0.3 * keyword_score)

            retrieved_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "final_score": final_score,
                }
            )

        retrieved_chunks.sort(key=lambda item: item["final_score"], reverse=True)
        return retrieved_chunks[:top_k]

    @staticmethod
    def build_context(retrieved_chunks: List[dict[str, Any]]) -> str:
        context_parts = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk["metadata"]
            title = metadata.get("title") or metadata.get("file_name") or "Unknown source"
            post_url = metadata.get("post_url") or "N/A"

            context_parts.append(
                f"[Source {index}] Title: {title}\n"
                f"URL: {post_url}\n"
                f"Content: {chunk['text']}"
            )

        return "\n\n".join(context_parts)

    @staticmethod
    def _semantic_score_from_distance(distance: float) -> float:
        # im mniejszy distance, tym lepiej
        return 1.0 / (1.0 + max(distance, 0.0))

    def _keyword_score(self, question: str, chunk_text: str) -> float:
        question_keywords = self._extract_keywords(question)
        chunk_lower = chunk_text.lower()

        if not question_keywords:
            return 0.0

        matched = 0
        total_weight = 0.0
        matched_weight = 0.0

        weighted_keywords = self._build_weighted_keywords(question_keywords)

        for keyword, weight in weighted_keywords:
            total_weight += weight
            if keyword in chunk_lower:
                matched += 1
                matched_weight += weight

        if total_weight == 0:
            return 0.0

        # score 0-1
        base_score = matched_weight / total_weight

        # mały bonus za liczbę dopasowań
        density_bonus = min(matched * 0.03, 0.15)

        return min(base_score + density_bonus, 1.0)

    @staticmethod
    def _extract_keywords(question: str) -> List[str]:
        question_lower = question.lower()

        # bazowe słowa z pytania
        raw_tokens = re.findall(r"\b[a-zA-Z0-9$]+\b", question_lower)

        stopwords = {
            "when", "did", "what", "was", "the", "and", "or", "a", "an",
            "of", "to", "in", "on", "for", "with", "is", "are"
        }

        keywords = [token for token in raw_tokens if token not in stopwords and len(token) > 2]

        # proste wzbogacenie semantyczne pod pytania faktograficzne
        if "go public" in question_lower or "went public" in question_lower:
            keywords.extend(["ipo", "public"])

        if "price per share" in question_lower:
            keywords.extend(["price", "share", "per share"])

        if "what price" in question_lower:
            keywords.extend(["price", "share"])

        # usunięcie duplikatów z zachowaniem kolejności
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return unique_keywords

    @staticmethod
    def _build_weighted_keywords(keywords: List[str]) -> List[tuple[str, float]]:
        weighted = []

        high_value_keywords = {
            "ipo": 2.0,
            "public": 1.5,
            "price": 1.7,
            "share": 1.7,
            "per share": 2.0,
        }

        for keyword in keywords:
            weight = high_value_keywords.get(keyword, 1.0)
            weighted.append((keyword, weight))

        return weighted