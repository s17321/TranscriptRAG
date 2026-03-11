#Fabryka tworząca odpowiedni klient na podstawie konfiguracji.

from .base import LLMClient
from .inference_api_client import build_inference_api_client


def get_llm_client() -> LLMClient:
    return build_inference_api_client()