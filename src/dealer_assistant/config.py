from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuration read from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: SecretStr = Field(min_length=1)
    gemini_chat_model: str = "gemini-3.5-flash-lite"
    stock_path: Path = ROOT / "data" / "stock_sample.csv"
    private_dir: Path = ROOT / "data" / "private"
    request_timeout_s: int = (
        20  # chat replies are short; a stalled call should be retried
    )
    max_output_tokens: int = 800
    llm_max_retries: int = 3
