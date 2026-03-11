#Konfiguracja endpointu, klucza, timeoutów, modelu domyślnego.

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    INFERENCE_API_URL: str
    OPENAI_API_KEY: str
    DEFAULT_LLM_MODEL: str = "generative-apis/devstral-2-123b-instruct-2512"
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_VERIFY_SSL: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()