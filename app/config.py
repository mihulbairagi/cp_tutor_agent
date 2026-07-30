"""Centralised application configuration.

Every tunable value in the project — handles, API keys, TTLs, retry counts,
scoring weights — is declared here as a typed field and read from the
environment (or a `.env` file). Nothing is hardcoded at the call site.

Why pydantic-settings rather than bare `os.getenv`:

* **Validation at startup.** A malformed `SYNC_INTERVAL_MINUTES=abc` fails
  immediately with a clear error instead of blowing up hours later inside the
  scheduler.
* **Typed access.** `settings.http_max_retries` is an `int`, not a `str` you
  have to remember to cast.
* **One object to inject.** Tests override a single `Settings` instance rather
  than monkeypatching scattered environment reads.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingBackend = Literal["sentence-transformers", "chroma-default"]
ChromaMode = Literal["persistent", "http"]
CacheBackend = Literal["sqlite", "memory"]


class Settings(BaseSettings):
    """Typed application settings, loaded from environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Identity ----------------------------------------------------------
    app_name: str = "CP Tutor Agent"
    app_version: str = "0.1.0"

    codeforces_handle: str = Field(default="", description="Codeforces handle to coach.")
    leetcode_username: str = Field(default="", description="LeetCode username to coach.")
    leetcode_session: str = Field(
        default="",
        description=(
            "Optional LEETCODE_SESSION cookie. When present the client can read the "
            "full solved list instead of only the ~20 most recent accepted submissions."
        ),
    )

    # --- LLM ---------------------------------------------------------------
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # --- Vector store ------------------------------------------------------
    chroma_mode: ChromaMode = "persistent"
    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    chroma_collection: str = "solved_problems"

    embedding_backend: EmbeddingBackend = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Cache -------------------------------------------------------------
    cache_backend: CacheBackend = "sqlite"
    cache_db_path: Path = Path("./data/cache.db")
    cache_ttl_seconds: int = 3_600
    # The Codeforces problemset is a ~1.5 MB payload that changes only when new
    # contests are added, so it gets a much longer TTL than user submissions.
    problemset_cache_ttl_seconds: int = 86_400

    # --- HTTP resilience ---------------------------------------------------
    http_timeout_seconds: float = 20.0
    http_max_retries: int = 4
    http_backoff_base_seconds: float = 0.5
    http_user_agent: str = "cp-tutor-agent/0.1 (+https://github.com/mihulbairagi/cp-tutor-agent)"

    # --- Scheduled sync ----------------------------------------------------
    sync_enabled: bool = True
    sync_interval_minutes: int = 360
    sync_on_startup: bool = False

    # --- Analysis tuning ---------------------------------------------------
    # Weights for the four weakness signals. Kept in config (not buried in the
    # scorer) so the ranking can be retuned without touching code.
    weight_coverage: float = 0.35
    weight_staleness: float = 0.25
    weight_difficulty: float = 0.20
    weight_struggle: float = 0.20

    target_solves_per_topic: int = 10
    staleness_horizon_days: int = 120
    min_topic_frequency: int = 40  # ignore rare Codeforces tags as "blind spots"

    # --- Recommendations ---------------------------------------------------
    rating_band_below: int = 100
    rating_band_above: int = 300
    default_rating_when_unrated: int = 1200
    max_problems_per_topic: int = 2

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    # --- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return level

    @field_validator("llm_temperature")
    @classmethod
    def _check_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("llm_temperature must be between 0.0 and 1.0")
        return value

    @property
    def llm_enabled(self) -> bool:
        """True when an Anthropic key is configured.

        The whole app degrades gracefully without one: the router falls back to
        a deterministic keyword classifier and template answers. That keeps CI
        and offline demos working without burning tokens.
        """
        return bool(self.anthropic_api_key.strip())

    @property
    def leetcode_authenticated(self) -> bool:
        """True when a session cookie is available for full LeetCode history."""
        return bool(self.leetcode_session.strip())

    @property
    def weakness_weights(self) -> dict[str, float]:
        return {
            "coverage": self.weight_coverage,
            "staleness": self.weight_staleness,
            "difficulty": self.weight_difficulty,
            "struggle": self.weight_struggle,
        }

    def ensure_data_dirs(self) -> None:
        """Create the directories the cache and vector store write into."""
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.chroma_mode == "persistent":
            self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that reading configuration is free at any call site, and so that
    FastAPI can use it directly as a dependency.
    """
    return Settings()
