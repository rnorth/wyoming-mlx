# Streaming STT via WhisperLiveKit — Design

**Date:** 2026-06-12
**Status:** Approved

## Goal

Replace the batch `mlx_whisper` transcription backend with
[WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) to get true
streaming speech-to-text: transcription happens while audio arrives, partial
results stream to clients, and the final transcript is available almost
immediately at end of speech.

## Background / engine selection

The original candidate was Lightning-SimulWhisper (altalt-org). Research ruled
it out: it is PolyForm Noncommercial licensed (incompatible with our
Apache-2.0 + PyPI distribution) and is not a pip-installable package.

WhisperLiveKit was selected instead:

- Apache-2.0, published on PyPI (pin a version; project is 0.x and moves fast)
- Implements the same AlignAtt SimulStreaming policy (upstream
  ufal/SimulStreaming relicensed to MIT in Oct 2025)
- First-class `mlx-whisper` backend for Apple Silicon
- Usable as a Python library: `TranscriptionEngine` (singleton, holds the
  model) + `AudioProcessor` (per-connection session)
- `pcm_input=True` accepts raw s16le mono 16 kHz PCM directly — no ffmpeg

Scope decision: **full switch, one engine**. Both the Wyoming STT server and
the HTTP `/v1/audio/transcriptions` endpoint use WhisperLiveKit. The
`mlx-whisper` direct dependency and `backends/mlx_stt.py` are removed.

## Architecture

### Backend protocol (`backends/base.py`)

`STTBackend.transcribe()` is replaced by a session-based streaming protocol:

```python
class STTBackend(Protocol):
    def start_session(self) -> STTSession: ...

class STTSession(Protocol):
    async def feed(self, audio: bytes, sample_rate: int) -> None: ...
    async def finish(self) -> None: ...
    def updates(self) -> AsyncIterator[STTUpdate]: ...
```

- `feed` accepts int16 mono PCM at any sample rate; resampling to 16 kHz is
  our responsibility (the existing `_resample_to_16k` helper is reused).
- `updates()` yields incremental updates while audio streams and terminates
  after the final transcript. Update shape:

  ```python
  @dataclass
  class STTUpdate:
      text: str            # newly confirmed text (never retracted)
      final: str | None    # full transcript; set only on the last update
  ```

### WhisperLiveKit backend (`backends/wlk_stt.py`)

- One `TranscriptionEngine` created **eagerly at startup** (also resolves the
  existing TODO about lazy STT model loading; fail fast if the model can't
  load). Configured with `backend="mlx-whisper"`, the default
  `simulstreaming` policy, `pcm_input=True`, and WLK's default VAD/VAC
  settings — the `--integration` test validates these on short utterances
  and they only become config knobs if that test shows they need tuning.
- Each `start_session()` wraps a fresh
  `AudioProcessor(transcription_engine=...)`:
  - `feed` → resample → `process_audio(bytes)`
  - `finish` → `process_audio(None)` (flush + end-of-stream)
  - `updates()` consumes the `FrontData` results generator: `lines` =
    confirmed segments (emitted as chunks), `buffer_transcription` =
    revisable tail (NOT emitted — Wyoming clients never see retractions).
    Final update carries the full joined text.

### Wyoming handler (`wyoming_servers/stt.py`)

Replaces buffer-then-transcribe:

- `AudioStart` → open session, start a task consuming `updates()`
- `AudioChunk` → `feed` (cumulative per-utterance byte cap retained)
- updates task emits `TranscriptStart` then `TranscriptChunk` events as
  confirmed text arrives
- `AudioStop` → `finish()`, await final update, emit final `Transcript`
  (backward compatible — pre-2025.7 HA reads only this) then `TranscriptStop`
- `Info` block sets `supports_transcript_streaming=True`
- Errors during a session are logged; the handler recovers to a clean state
  (matching current behaviour)

### HTTP endpoint (`http/routes.py`)

Contract unchanged. Implementation: decode upload → one session → feed all →
finish → drain updates → return `{"text": final}`.

### Config (`config.py`)

- `models.whisper` keeps its name; its value becomes a WhisperLiveKit model
  name. Default changes `mlx-community/distil-whisper-large-v3` →
  `large-v3-turbo` (distil checkpoints lack the alignment heads AlignAtt
  needs; large-v3-turbo runs real-time on Apple Silicon).
- `_ensure_models_cached` whisper branch replaced by the engine's eager init.

### Dependencies (`pyproject.toml`)

- Add `whisperlivekit[mlx-whisper]; sys_platform == 'darwin'` (pinned)
- Remove `mlx-whisper`
- Only the library layer is used; WLK's server/web UI/diarization stay unused

## Error handling

- Engine init failure at startup → log and exit non-zero (fail fast)
- Session failure mid-utterance → log exception, send empty/partial final
  transcript per current semantics, reset handler state
- Oversize utterance → same warning + reset as today, applied to the
  cumulative counter

## Testing

- `FakeSTTBackend` reimplements the session protocol with scripted partial +
  final texts → handler streaming behaviour (event ordering, chunk content,
  back-compat final `Transcript`) unit-tested on Linux CI
- HTTP route tests updated to the session protocol via the fake
- Roundtrip integration test asserts `TranscriptStart` → `TranscriptChunk*`
  → `Transcript` → `TranscriptStop` ordering
- Real-engine test behind `--integration` on Apple Silicon; must validate
  short-utterance behaviour (VAD/timing on voice-assistant-length audio)

## Risks

- WhisperLiveKit is 0.x: pin the version; expect API churn on upgrades
- `TranscriptionEngine` is a process-wide singleton: one model/config per
  process (already true today); tests use `TranscriptionEngine.reset()`
- Short-utterance quality/latency under AlignAtt + VAD needs empirical
  validation on the Mac Studio before merge
