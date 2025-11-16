"""
Configuration management for the LLM Classification System.
Uses environment variables with sensible defaults.
"""

from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for the LLM model."""
    name: str = os.getenv("MODEL_NAME", "deepseek-r1-distill-llama-70b")
    input_cost_per_million: float = float(os.getenv("INPUT_COST_PER_MILLION", "0.15"))
    output_cost_per_million: float = float(os.getenv("OUTPUT_COST_PER_MILLION", "0.60"))
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.0"))
    max_retries: int = int(os.getenv("MODEL_MAX_RETRIES", "3"))
    token_count_model: str = os.getenv("TOKEN_COUNT_MODEL", "gpt-3.5-turbo")


@dataclass
class ChromaConfig:
    """Configuration for ChromaDB."""
    path: str = os.getenv("CHROMA_DB_PATH", "my_vectordb")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
    interaction_collection: str = os.getenv("INTERACTION_COLLECTION", "customer_interaction")
    policies_collection: str = os.getenv("POLICIES_COLLECTION", "customer_policies")
    reset_on_startup: bool = os.getenv("RESET_COLLECTIONS", "false").lower() == "true"
    query_n_results: int = int(os.getenv("CHROMA_QUERY_N_RESULTS", "1"))


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig = None
    chroma: ChromaConfig = None
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.chroma is None:
            self.chroma = ChromaConfig()


# Global configuration instance
config = AppConfig()

