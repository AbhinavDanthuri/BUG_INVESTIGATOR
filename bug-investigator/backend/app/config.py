from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent  # .../backend
_PROJECT_ROOT = _BACKEND_DIR.parent

# Look for .env in the project root and in backend/, so it is found whichever directory you
# run from. Later files win, and real environment variables always beat both (Docker sets them).
_ENV_FILES = (_PROJECT_ROOT / ".env", _BACKEND_DIR / ".env", Path(".env"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILES, extra="ignore")

    database_url: str = "mysql+pymysql://bugi:bugi@localhost:3306/bug_investigator"
    github_token: str = ""

    llm_provider: str = "gemini"  # gemini | groq | anthropic | openai | ollama
    llm_model: str = "gemini-2.5-flash"  # check AI Studio for the current Flash model name
    google_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: str = "./data"

    max_iterations: int = 3
    token_budget: int = 200_000
    retrieval_top_k: int = 8
    prompt_version: str = "v2"  # v2 added the is_defect verdict; runs are not comparable across versions


@lru_cache
def get_settings() -> Settings:
    return Settings()
