#Interfejs klienta LLM.

from abc import ABC, abstractmethod
from .models import GenerationRequest, GenerationResponse


class LLMClient(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response from the language model."""
        raise NotImplementedError