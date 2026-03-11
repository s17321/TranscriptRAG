from typing import Any, Optional

from app.infrastructure.llm.base import LLMClient
from app.infrastructure.llm.models import ChatMessage, GenerationRequest
from app.services.retrieval_service import RetrievalService


class ChatService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_client: LLMClient,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client

    def ask(
        self,
        question: str,
        top_k: int = 5,
        file_name: Optional[str] = None,
    ) -> dict[str, Any]:

        filters = None
        if file_name:
            filters = {"file_name": file_name}

        retrieved_chunks = self.retrieval_service.retrieve(
            question=question,
            top_k=top_k,
            filters=filters,
        )

        context = self.retrieval_service.build_context(retrieved_chunks)

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a helpful assistant answering questions about podcast transcripts. "
                    "Use only the provided context. "
                    "Extract concrete facts carefully, including dates, numbers, and prices when available. "
                    "If the answer is not present in the context, say that you do not know. "
                    "When possible, give a concise but complete answer."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            ),
        ]

        request = GenerationRequest(
            messages=messages,
            temperature=0.1,
            max_tokens=700,
        )

        response = self.llm_client.generate(request)

        return {
            "question": question,
            "answer": response.content,
            "sources": retrieved_chunks,
            "model": response.model,
            "usage": response.usage,
            "file_name_filter": file_name,
        }