from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_overrides(prefix: str, delimiter: str = "__") -> dict:
    """Extract environment variables matching *prefix* and return as nested dict.

    E.g. WYOMING_MLX_HTTP__PORT=13333 -> {"http": {"port": "13333"}}
    """
    out: dict = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        if not remainder:
            continue
        parts = remainder.strip(delimiter).split(delimiter)
        d = out
        for part in parts[:-1]:
            d = d.setdefault(part.lower(), {})
        d[parts[-1].lower()] = value
    return out


class ModelsConfig(BaseModel):
    whisper: str = "large-v3-turbo"
    kokoro: str = "hexgrad/Kokoro-82M"
    kokoro_default_voice: str = "af_heart"


class WyomingConfig(BaseModel):
    stt_host: str = "127.0.0.1"
    stt_port: int = 10300
    tts_host: str = "127.0.0.1"
    tts_port: int = 10200
    stt_max_audio_bytes: int = Field(default=100_000_000, ge=1)


class HttpConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 10400
    api_keys_file: str = "~/.config/wyoming-mlx/apikeys"


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseSettings):
    """Top-level configuration.

    Precedence: CLI > env (WYOMING_MLX_*) > TOML file > defaults.
    Nested fields use double-underscore in env, e.g. WYOMING_MLX_HTTP__PORT.
    """

    model_config = SettingsConfigDict(
        env_prefix="WYOMING_MLX_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    wyoming: WyomingConfig = Field(default_factory=WyomingConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Path | None) -> Config:
    """Load configuration: TOML file (if given) -> overridden by env vars -> CLI."""
    toml_dict: dict = {}
    if config_path is not None:
        try:
            with config_path.open("rb") as f:
                toml_dict = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"malformed TOML in {config_path}: {exc}") from exc

    # Env vars take precedence over TOML values.
    # We build a merged dict because pydantic-settings does *not*
    # override explicit constructor kwargs with env vars.
    env_dict = _env_overrides("WYOMING_MLX_")
    merged = _deep_merge(toml_dict, env_dict)
    return Config(**merged)


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with *override* values recursively merged into *base*."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
