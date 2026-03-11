#Modele wejścia/wyjścia dla warstwy LLM.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class GenerationRequest:
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1000
    top_p: float = 1.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    content: str
    model: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None