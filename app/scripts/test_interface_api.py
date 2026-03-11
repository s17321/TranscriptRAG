from app.infrastructure.llm.inference_api_client import build_inference_api_client
from app.infrastructure.llm.models import ChatMessage, GenerationRequest


def main() -> None:
    client = build_inference_api_client()

    request = GenerationRequest(
        model="generative-apis/devstral-2-123b-instruct-2512",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Say hello in one short sentence."),
        ],
        temperature=0.1,
        max_tokens=50,
    )

    response = client.generate(request)
    print("MODEL:", response.model)
    print("ANSWER:", response.content)


if __name__ == "__main__":
    main()