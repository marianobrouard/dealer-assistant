import pytest
from pydantic import ValidationError

from dealer_assistant.config import Settings


def test_settings_defaults_favor_a_short_timeout_for_interactive_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-a-real-key")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]  # key comes from env
    assert settings.request_timeout_s <= 30
    assert settings.llm_max_retries <= 3


def test_settings_with_empty_api_key_fails_early_naming_the_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    with pytest.raises(ValidationError, match="google_api_key"):
        Settings(_env_file=None)  # type: ignore[call-arg]  # key is empty on purpose
