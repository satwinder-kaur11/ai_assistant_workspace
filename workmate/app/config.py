"""
app/config.py

Loads environment variables (like API keys) from the .env file using Pydantic Settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # 'anthropic' or 'ollama'
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # 'local' or 'openai'
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DB_URL = os.getenv("DB_URL", "sqlite:///./workmate.db")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    MEMORY_IMPORTANCE_THRESHOLD = float(os.getenv("MEMORY_IMPORTANCE_THRESHOLD", "0.7"))
    RAG_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.3"))

settings = Settings()
# loads the environment variables from the .env file and creates an instance of the Settings class,
#  which can be imported and used throughout the application to access configuration values.
