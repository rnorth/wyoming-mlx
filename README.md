# wyoming-mlx

Apple-Silicon-native TTS (Kokoro) and streaming STT (Whisper via WhisperLiveKit) for Home Assistant
and OpenAI-compatible clients.

## Why?

If you have a Mac on your network, it can be your voice server. wyoming-mlx
turns it into a fast, fully local speech-to-text and text-to-speech service:

- **Private by construction.** Audio never leaves your network — no cloud
  speech APIs, no per-request pricing, nothing to subscribe to. Models run
  entirely on your machine.
- **Uses hardware you already own.** Apple Silicon's GPU and unified memory
  run Whisper and Kokoro comfortably alongside whatever else the Mac is
  doing — no dedicated GPU server, no idle power draw of a CUDA box.
- **One service, two ecosystems.** Home Assistant talks to it natively over
  the [Wyoming protocol](https://github.com/rhasspy/wyoming) (drop-in
  replacement for `wyoming-faster-whisper`/`wyoming-piper` satellites), while
  anything that speaks the OpenAI audio API — scripts, editors, chat UIs —
  can use the same instance via `/v1/audio/transcriptions` and
  `/v1/audio/speech`.
- **Set-and-forget.** Install with Homebrew, run as a launchd service via
  `brew services`, and models are fetched once into the Hugging Face cache.

Speech-to-text streams through [WhisperLiveKit](https://github.com/collabora/WhisperLiveKit)'s
SimulStreaming (AlignAtt) policy on MLX, so partial transcripts arrive while
you're still speaking; text-to-speech runs Kokoro on Metal via PyTorch. The
real backends therefore require an Apple Silicon Mac. Everything else (config,
HTTP API, Wyoming protocol handling, fake backends, tests) is portable, and
CI runs on Linux against the fake backends.

## Install (Homebrew)

```bash
brew tap rnorth/tap
brew install wyoming-mlx
```

Run it in the foreground with `wyoming-mlx`, or as a launchd service:

```bash
brew services start wyoming-mlx
```

Logs go to `$(brew --prefix)/var/log/wyoming-mlx.log`. Apple Silicon only.

> [!WARNING]
> **Upgraders:** the `models.whisper` config value (TOML `[models] whisper =
> ...`, CLI `--whisper-model`, env `WYOMING_MLX_MODELS__WHISPER`) changed
> meaning in this release. It used to be a Hugging Face repo ID (e.g.
> `mlx-community/distil-whisper-large-v3`); it is now a WhisperLiveKit
> model-size name: `tiny`, `base`, `small`, `medium`, `large-v3`, or
> `large-v3-turbo` (default `large-v3-turbo`). An old repo-ID value (anything
> containing `/`) is rejected at startup with a clear error. Update your
> config, for example:
>
> ```toml
> [models]
> # before:  whisper = "mlx-community/distil-whisper-large-v3"
> whisper = "large-v3-turbo"
> ```

## Quick start (dev)

```bash
mise install
uv sync
uv run pytest
```

Works on Linux too: CPU-only torch is selected automatically (the fake
backends need no GPU). Integration tests against real models are skipped by
default; run them with `uv run pytest --integration` (Apple Silicon only).

## Run locally (fake backends)

```bash
uv run python scripts/dev_run.py
```

The dev server uses the API key `dev`.

## Run locally (real MLX backends)

```bash
uv run wyoming-mlx
```

By default it loads:
- whisper-large-v3-turbo (MLX, streaming) on Wyoming port 10300 / HTTP `/v1/audio/transcriptions`
- Kokoro-82M (MLX) on Wyoming port 10200 / HTTP `/v1/audio/speech`
- HTTP on port 10400 with API-key auth

Models are stored in the Hugging Face cache. The STT model is loaded at
startup (so a bad model name or download failure surfaces immediately); the
Kokoro TTS model downloads on first synthesis.

### API keys

HTTP endpoints require a bearer token. Keys are read at startup from
`~/.config/wyoming-mlx/apikeys` (override with `--http-api-keys-file`),
one key per line, `#` comments allowed. The file should be mode `0600`.
If the file is missing or empty, all HTTP requests are rejected with 401.

Note that the HTTP API listens on all interfaces by default (set
`WYOMING_MLX_HTTP__HOST=127.0.0.1` to restrict it), and `GET /v1/models`
is unauthenticated, matching OpenAI API behaviour.

```bash
mkdir -p ~/.config/wyoming-mlx
(umask 077; openssl rand -hex 32 > ~/.config/wyoming-mlx/apikeys)
```

## HTTP API

### List models

```bash
curl http://localhost:10400/v1/models
```

### Transcribe an audio file

```bash
curl http://localhost:10400/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@some.wav
```

### Synthesize speech

```bash
curl http://localhost:10400/v1/audio/speech \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello there.","voice":"af_heart"}' \
  --output /tmp/out.wav
```

## Home Assistant integration

Settings → Integrations → Wyoming Protocol → Add:

- STT: `<host>:10300`
- TTS: `<host>:10200`

No keys, no TLS (HA convention, trusted LAN).

Streaming transcripts (live partial results) require Home Assistant 2025.7 or
newer; older versions still receive the final transcript exactly as before.

## Configuration

Pass `--config /path/to/config.toml` or set env vars with the
`WYOMING_MLX_` prefix and `__` for nesting (e.g.
`WYOMING_MLX_HTTP__PORT=10401`). See `src/wyoming_mlx/config.py` for the
full schema.

## License

[Apache-2.0](LICENSE)
