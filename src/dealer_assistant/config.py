from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class StockSettings(BaseSettings):
    """Only the stock location: lets read-only tools run without any model key."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    stock_path: Path = ROOT / "data" / "stock_sample.csv"


class Settings(StockSettings):
    """Configuration read from environment variables or a local .env file."""

    google_api_key: SecretStr = Field(min_length=1)
    gemini_chat_model: str = "gemini-3.5-flash-lite"
    private_dir: Path = ROOT / "data" / "private"
    request_timeout_s: int = (
        20  # chat replies are short; a stalled call should be retried
    )
    max_output_tokens: int = 800
    llm_max_retries: int = 3
