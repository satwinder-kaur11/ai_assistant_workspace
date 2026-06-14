import logging
from app.config import settings

logger = logging.getLogger(__name__)

def get_llm(temperature: float = 0):
    """
    Returns a configured LLM instance based on LLM_PROVIDER setting.
    Supports 'anthropic' and 'ollama'.
    """
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        logger.info(f"Using local Ollama model: {settings.OLLAMA_MODEL}")
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            temperature=temperature
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not settings.ANTHROPIC_API_KEY or settings.ANTHROPIC_API_KEY == "your-anthropic-api-key-here":
            raise ValueError("ANTHROPIC_API_KEY must be set when LLM_PROVIDER is 'anthropic'")
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=4096,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

def has_active_llm() -> bool:
    """
    Checks if a valid LLM provider is configured.
    """
    provider = settings.LLM_PROVIDER.lower()
    if provider == "ollama":
        return True
    elif provider == "anthropic":
        key = (settings.ANTHROPIC_API_KEY or "").strip()
        return bool(key) and key != "your-anthropic-api-key-here"
    return False
