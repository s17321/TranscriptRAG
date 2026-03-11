from app.infrastructure.llm.base import LLMClient
from app.infrastructure.llm.models import ChatMessage, GenerationRequest


class ChatService:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def ask(self, user_question: str, context: str) -> str:
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a helpful assistant. "
                    "Answer only based on the provided context. "
                    "If the answer is not in the context, say you do not know."
                ),
            ),
            ChatMessage(
                role="user",
                content=f"Context:\n{context}\n\nQuestion:\n{user_question}",
            ),
        ]

        request = GenerationRequest(
            messages=messages,
            temperature=0.1,
            max_tokens=800,
        )

        response = self.llm_client.generate(request)
        return response.content