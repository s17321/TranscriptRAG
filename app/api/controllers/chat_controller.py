from fastapi import APIRouter, HTTPException

from app.api.schemas.chat_schema import ChatAskRequest, ChatAskResponse
from app.infrastructure.llm.inference_api_client import build_inference_api_client
from app.repositories.vector_repository import VectorRepository
from app.services.chat_service import ChatService
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/chat", tags=["chat"])

# Prosty singleton na poziomie modułu, żeby nie inicjalizować wszystkiego przy każdym request
_embedding_service = EmbeddingService()
_vector_repository = VectorRepository()
_retrieval_service = RetrievalService(
    vector_repository=_vector_repository,
    embedding_service=_embedding_service,
)
_llm_client = build_inference_api_client()
_chat_service = ChatService(
    retrieval_service=_retrieval_service,
    llm_client=_llm_client,
)


@router.post("/ask", response_model=ChatAskResponse)
def ask_chat(request: ChatAskRequest) -> ChatAskResponse:
    try:
        result = _chat_service.ask(
            question=request.question,
            top_k=request.top_k,
            file_name=request.file_name,
        )
        return ChatAskResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG request failed: {exc}") from exc