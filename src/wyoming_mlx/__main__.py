"""Entry point: start Wyoming STT, Wyoming TTS, and HTTP servers."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import contextlib
import logging
import multiprocessing
import os
import sys
from pathlib import Path

import uvicorn
from wyoming.info import (
    AsrModel,
    AsrProgram,
    Attribution,
    Info,
    TtsProgram,
    TtsVoice,
)
from wyoming.server import AsyncServer

from wyoming_mlx.auth import load_api_keys
from wyoming_mlx.backends.base import STTBackend, TTSBackend
from wyoming_mlx.config import Config, load_config
from wyoming_mlx.http.app import create_app
from wyoming_mlx.wyoming_servers.stt import SttEventHandler
from wyoming_mlx.wyoming_servers.tts import TtsEventHandler

log = logging.getLogger(__name__)


def _ensure_models_cached(kokoro_model_id: str) -> None:
    """One-time lazy download of the Kokoro TTS model at startup.

    huggingface_hub contacts the HF API to resolve revision tags on every
    call, but once files are in the local cache subsequent loads are
    completely offline.  This function ensures the cache is populated at
    startup so the first TTS request doesn't block.

    WhisperLiveKit downloads its model during eager backend initialisation,
    so no pre-caching is needed for STT.

    If the model is already cached (we check via a marker file), this is
    a no-op.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    marker = hf_home / ".wyoming-mlx-cached"
    cache_key = kokoro_model_id

    # Quick check without importing heavy modules.
    if marker.exists():
        try:
            if marker.read_text().strip() == cache_key:
                return
        except OSError:
            pass

    log.info("Checking HuggingFace model cache …")

    def _download() -> None:
        from huggingface_hub import hf_hub_download

        # Kokoro model.
        try:
            hf_hub_download(
                kokoro_model_id,
                "config.json",
                cache_dir=str(hf_home),
            )
            # Determine the model file name from KModel.MODEL_NAMES.
            try:
                from kokoro.model import KModel

                model_file = KModel.MODEL_NAMES.get(kokoro_model_id, "kokoro-v1_0.pth")
            except ImportError:
                model_file = "kokoro-v1_0.pth"
            hf_hub_download(
                kokoro_model_id,
                model_file,
                cache_dir=str(hf_home),
            )
            log.info("Kokoro model cached.")
        except Exception:
            log.warning("Failed to cache Kokoro model (will download on first use).")

    try:
        _download()
        marker.write_text(cache_key)
    except Exception:
        log.warning("Model cache check failed — models will download lazily.")


_PROJECT_URL = "https://github.com/rnorth/wyoming-mlx"


def _build_stt_info(model_id: str) -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="wyoming-mlx",
                attribution=Attribution(name="wyoming-mlx", url=_PROJECT_URL),
                installed=True,
                description="WhisperLiveKit streaming STT (MLX)",
                version="0.1.0",
                supports_transcript_streaming=True,
                models=[
                    AsrModel(
                        name=model_id,
                        attribution=Attribution(name="OpenAI/MLX", url=_PROJECT_URL),
                        installed=True,
                        description=model_id,
                        version="0.1.0",
                        languages=["en"],
                    )
                ],
            )
        ],
    )


def _build_tts_info(voices: list[str]) -> Info:
    return Info(
        tts=[
            TtsProgram(
                name="wyoming-mlx",
                attribution=Attribution(name="wyoming-mlx", url=_PROJECT_URL),
                installed=True,
                description="MLX Kokoro TTS",
                version="0.1.0",
                voices=[
                    TtsVoice(
                        name=v,
                        attribution=Attribution(name="Kokoro", url=_PROJECT_URL),
                        installed=True,
                        description=v,
                        version="0.1.0",
                        languages=["en"],
                    )
                    for v in voices
                ],
            )
        ],
    )


async def run_servers(
    *,
    cfg: Config,
    stt: STTBackend,
    tts: TTSBackend,
    api_keys: set[str],
) -> None:
    """Start Wyoming STT, Wyoming TTS, and HTTP servers and run until cancelled."""

    stt_info = _build_stt_info(cfg.models.whisper)
    tts_info = _build_tts_info(tts.voices)

    stt_server = AsyncServer.from_uri(f"tcp://{cfg.wyoming.stt_host}:{cfg.wyoming.stt_port}")
    tts_server = AsyncServer.from_uri(f"tcp://{cfg.wyoming.tts_host}:{cfg.wyoming.tts_port}")

    app = create_app(
        stt=stt,
        tts=tts,
        api_keys=api_keys,
        models=cfg.models,
    )
    http_config = uvicorn.Config(
        app,
        host=cfg.http.host,
        port=cfg.http.port,
        log_level=cfg.logging.level.lower(),
    )
    http_server = uvicorn.Server(http_config)

    log.info(
        "Starting wyoming-mlx (STT=%s:%s TTS=%s:%s HTTP=%s:%s)",
        cfg.wyoming.stt_host,
        cfg.wyoming.stt_port,
        cfg.wyoming.tts_host,
        cfg.wyoming.tts_port,
        cfg.http.host,
        cfg.http.port,
    )

    stt_server_task = asyncio.create_task(
        stt_server.run(
            lambda reader, writer: SttEventHandler(
                reader,
                writer,
                backend=stt,
                info=stt_info,
                max_audio_bytes=cfg.wyoming.stt_max_audio_bytes,
            ),
        ),
        name="stt-server",
    )
    tts_server_task = asyncio.create_task(
        tts_server.run(
            lambda reader, writer: TtsEventHandler(
                reader,
                writer,
                backend=tts,
                default_voice=cfg.models.kokoro_default_voice,
                info=tts_info,
            ),
        ),
        name="tts-server",
    )
    http_server_task = asyncio.create_task(
        http_server.serve(),
        name="http-server",
    )

    await asyncio.gather(stt_server_task, tts_server_task, http_server_task)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="wyoming-mlx: MLX STT/TTS server")
    parser.add_argument("--config", type=Path, default=None, help="Path to TOML config file")
    parser.add_argument("--stt-host", default=None)
    parser.add_argument("--stt-port", type=int, default=None)
    parser.add_argument("--tts-host", default=None)
    parser.add_argument("--tts-port", type=int, default=None)
    parser.add_argument("--http-host", default=None)
    parser.add_argument("--http-port", type=int, default=None)
    parser.add_argument("--http-api-keys-file", default=None)
    parser.add_argument("--whisper-model", default=None)
    parser.add_argument("--kokoro-model", default=None)
    parser.add_argument("--kokoro-default-voice", default=None)
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def _apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> None:
    """Override config values with CLI arguments (non-None)."""
    overrides: list[tuple[str, str, object]] = [
        ("wyoming.stt_host", "stt_host", args.stt_host),
        ("wyoming.stt_port", "stt_port", args.stt_port),
        ("wyoming.tts_host", "tts_host", args.tts_host),
        ("wyoming.tts_port", "tts_port", args.tts_port),
        ("http.host", "http_host", args.http_host),
        ("http.port", "http_port", args.http_port),
        ("http.api_keys_file", "http_api_keys_file", args.http_api_keys_file),
        ("models.whisper", "whisper_model", args.whisper_model),
        ("models.kokoro", "kokoro_model", args.kokoro_model),
        ("models.kokoro_default_voice", "kokoro_default_voice", args.kokoro_default_voice),
        ("logging.level", "log_level", args.log_level),
    ]
    for dotpath, _attr, value in overrides:
        if value is not None:
            obj = cfg
            parts = dotpath.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the wyoming-mlx console script."""
    args = _parse_args(argv)
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    api_keys = load_api_keys(cfg.http.api_keys_file)

    # Lazy import real backends only at runtime
    try:
        from wyoming_mlx.backends.mlx_tts import KokoroBackend
        from wyoming_mlx.backends.wlk_stt import WhisperLiveKitBackend
    except ImportError:
        log.error(
            "MLX backends not available (mlx, whisperlivekit, kokoro packages missing). "
            "Install the MLX dependencies to enable real models."
        )
        sys.exit(1)

    # Ensure model weights are cached locally before backends initialize.
    _ensure_models_cached(cfg.models.kokoro)

    stt_backend = WhisperLiveKitBackend(model=cfg.models.whisper)
    tts_backend = KokoroBackend(model_id=cfg.models.kokoro, voice=cfg.models.kokoro_default_voice)

    # Suppress misaki's leaked semaphore warning at shutdown (multiprocessing
    # resource tracker retains a semaphore that is never closed).
    def _cleanup_mp() -> None:
        with contextlib.suppress(Exception):
            multiprocessing.resource_tracker._resource_tracker.shutdown()  # type: ignore[attr-defined]

    atexit.register(_cleanup_mp)

    asyncio.run(run_servers(cfg=cfg, stt=stt_backend, tts=tts_backend, api_keys=api_keys))


if __name__ == "__main__":
    main()
