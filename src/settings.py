import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    llm_provider: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    ollama_model: str
    ollama_base_url: str
    conversation_history_limit: int


def get_settings():
    return Settings(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://legal_agent:legal_agent@localhost:5432/legal_agent",
        ),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek").lower(),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        conversation_history_limit=int(os.getenv("CONVERSATION_HISTORY_LIMIT", "6")),
    )
