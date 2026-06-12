from pathlib import Path

import pytest

from wyoming_mlx.config import load_config


def test_defaults_when_no_file_no_env(monkeypatch: pytest.MonkeyPatch):
    for key in list(__import__("os").environ):
        if key.startswith("WYOMING_MLX_"):
            monkeypatch.delenv(key, raising=False)

    cfg = load_config(config_path=None)
    assert cfg.wyoming.stt_port == 10300
    assert cfg.wyoming.tts_port == 10200
    assert cfg.http.port == 10400
    assert cfg.models.whisper == "large-v3-turbo"
    assert cfg.logging.level == "INFO"


def test_toml_overrides_defaults(tmp_path: Path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
        [wyoming]
        stt_port = 11111

        [http]
        port = 12222
        """
    )
    cfg = load_config(config_path=cfg_file)
    assert cfg.wyoming.stt_port == 11111
    assert cfg.http.port == 12222
    # Untouched defaults remain.
    assert cfg.wyoming.tts_port == 10200


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[http]\nport = 12222\n")
    monkeypatch.setenv("WYOMING_MLX_HTTP__PORT", "13333")
    cfg = load_config(config_path=cfg_file)
    assert cfg.http.port == 13333


def test_malformed_toml_raises(tmp_path: Path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("this is not = valid = toml = [")
    with pytest.raises(ValueError, match="malformed TOML"):
        load_config(config_path=cfg_file)
