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
    ollama_timeout: int
    ollama_max_tokens: int
    deepseek_timeout: int
    conversation_history_limit: int
    demo_fast_mode: bool
    skip_query_rewrite: bool
    skip_rerank: bool
    fast_retrieval_top_k: int
    warmup_retrieval: bool


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        ollama_timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
        ollama_max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "1024")),
        deepseek_timeout=int(os.getenv("DEEPSEEK_TIMEOUT", "120")),
        conversation_history_limit=int(os.getenv("CONVERSATION_HISTORY_LIMIT", "6")),
        demo_fast_mode=_env_bool("DEMO_FAST_MODE", True),
        skip_query_rewrite=_env_bool("SKIP_QUERY_REWRITE", True),
        skip_rerank=_env_bool("SKIP_RERANK", True),
        fast_retrieval_top_k=int(os.getenv("FAST_RETRIEVAL_TOP_K", "3")),
        warmup_retrieval=_env_bool("WARMUP_RETRIEVAL", True),
    )
