from app.infrastructure.llm.inference_api_client import build_inference_api_client
from app.repositories.vector_repository import VectorRepository
from app.services.chat_service import ChatService
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService


def main() -> None:
    question = "When did Airbnb go public and what was the price per share?"

    embedding_service = EmbeddingService()
    vector_repository = VectorRepository()
    retrieval_service = RetrievalService(
        vector_repository=vector_repository,
        embedding_service=embedding_service,
    )
    llm_client = build_inference_api_client()
    chat_service = ChatService(
        retrieval_service=retrieval_service,
        llm_client=llm_client,
    )

    USE_FILE_NAME_FILTER = True
    FILE_NAME = "airbnb" if USE_FILE_NAME_FILTER else None

    result = chat_service.ask(
        question=question,
        top_k=10,
        file_name=FILE_NAME,
    )

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for index, source in enumerate(result["sources"], start=1):
        metadata = source["metadata"]
        print(
            f"{index}. "
            f"file_name={metadata.get('file_name')} | "
            f"title={metadata.get('title')} | "
            f"distance={source.get('distance')} | "
            f"semantic_score={source.get('semantic_score'):.4f} | "
            f"keyword_score={source.get('keyword_score'):.4f} | "
            f"final_score={source.get('final_score'):.4f}"
        )
        print(source["text"][:500])
        print("-" * 80)


if __name__ == "__main__":
    main()