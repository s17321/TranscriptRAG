#Konkretny adapter do Twojego wewnętrznego API.

import requests
from typing import Any, Dict

from app.infrastructure.config.settings import settings
from .base import LLMClient
from .models import GenerationRequest, GenerationResponse


class InferenceApiClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        timeout_seconds: int = 60,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout_seconds = timeout_seconds
        self.verify_ssl = verify_ssl

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        url = f"{self.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }

        if request.extra_params:
            payload.update(request.extra_params)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
            verify=self.verify_ssl,
        )
        response.raise_for_status()

        data = response.json()

        content = self._extract_content(data)

        return GenerationResponse(
            content=content,
            model=data.get("model"),
            raw_response=data,
            usage=data.get("usage"),
        )

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("LLM response does not contain choices.")

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content")

        if not content:
            raise ValueError("LLM response does not contain message content.")

        return content


def build_inference_api_client() -> InferenceApiClient:
    return InferenceApiClient(
        base_url=settings.INFERENCE_API_URL,
        api_key=settings.OPENAI_API_KEY,
        default_model=settings.DEFAULT_LLM_MODEL,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        verify_ssl=settings.LLM_VERIFY_SSL,
    )