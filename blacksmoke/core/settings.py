from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Secrets and deployment-specific values, read from the environment or `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: SecretStr | None = None
    openai_org_id: str | None = None
    google_api_key: SecretStr | None = None
    config_dir: Path = Field(
        default=PROJECT_ROOT / "config", validation_alias="BLACKSMOKE_CONFIG_DIR"
    )
