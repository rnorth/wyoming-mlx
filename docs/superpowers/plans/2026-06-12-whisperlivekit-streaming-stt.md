# Streaming STT via WhisperLiveKit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the batch `mlx_whisper` STT backend with WhisperLiveKit so transcription streams while audio arrives, emitting Wyoming `TranscriptStart`/`TranscriptChunk`/`Transcript`/`TranscriptStop` events.

**Architecture:** A session-based `STTBackend` protocol replaces the one-shot `transcribe()` call. The Wyoming handler opens a session per utterance, feeds chunks, and pumps incremental updates to the client. The HTTP batch endpoint reuses the same sessions via a `collect_transcript()` helper. One WhisperLiveKit `TranscriptionEngine` (singleton) is created eagerly at startup; each session wraps an `AudioProcessor`.

**Tech Stack:** Python 3.12, `whisperlivekit[mlx-whisper]==0.2.22` (AlignAtt SimulStreaming policy, `pcm_input=True` raw-PCM mode — no ffmpeg), `wyoming` (already supports streaming ASR events), pytest with `asyncio_mode = "auto"`.

**Spec:** `docs/superpowers/specs/2026-06-12-whisperlivekit-streaming-stt-design.md`

**Verified upstream facts (do not re-derive):**
- `TranscriptionEngine(**kwargs)` builds a `WhisperLiveKitConfig` from kwargs; valid kwargs include `model_size`, `backend`, `backend_policy`, `pcm_input`, `lan`, `log_level`. It is a thread-safe singleton; `TranscriptionEngine.reset()` exists for tests.
- `AudioProcessor(transcription_engine=engine)` is one streaming session. `results = await processor.create_tasks()` returns an async generator of `FrontData`. `await processor.process_audio(pcm_bytes)` feeds raw **s16le mono 16 kHz** PCM. `await processor.process_audio(None)` flushes and starts the stop sequence; the results generator then terminates. `await processor.cleanup()` cancels internal tasks.
- `FrontData` fields: `status` (`"active_transcription"`, `"no_audio_detected"`, or `"error"`), `error`, `lines` (list of `Segment`, each with optional `.text`), `buffer_transcription` (unconfirmed tail, str).
- wyoming `AsrProgram` has `supports_transcript_streaming: bool`; `TranscriptStart()`, `TranscriptChunk(text=...)`, `TranscriptStop()` exist in `wyoming.asr`. `AsyncEventHandler` has an `async def disconnect()` hook.
- `"large-v3-turbo"` maps to `mlx-community/whisper-large-v3-turbo` in WLK's model mapping.

**Conventions:** run everything with `uv run`. Line length 100 (ruff). `from __future__ import annotations` at top of modules. Tests are plain `async def` functions (asyncio auto mode).

---

### Task 1: Session protocol, STTUpdate, collect_transcript, fake sessions

Adds the new streaming protocol alongside the old `transcribe()` (removed in Task 3) so the suite stays green at every commit.

**Files:**
- Modify: `src/wyoming_mlx/backends/base.py`
- Modify: `src/wyoming_mlx/backends/fake.py`
- Test: `tests/unit/backends/test_fake.py`

- [ ] **Step 1: Write failing tests for the fake session and collect_transcript**

Append to `tests/unit/backends/test_fake.py`:

```python
from wyoming_mlx.backends.base import STTUpdate, collect_transcript


async def test_fake_stt_session_yields_partials_then_final():
    backend = FakeSTTBackend(transcript="hello world", partials=["hello ", "world"])
    session = backend.start_session()
    await session.feed(b"\x01\x02", 16000)
    await session.finish()

    updates = [u async for u in session.updates()]

    assert [u.text for u in updates[:-1]] == ["hello ", "world"]
    assert updates[-1] == STTUpdate(final="hello world")
    assert session.fed == [(b"\x01\x02", 16000)]


async def test_fake_stt_session_final_waits_for_finish():
    backend = FakeSTTBackend(transcript="done")
    session = backend.start_session()

    agen = session.updates()
    import asyncio

    pump = asyncio.ensure_future(anext(agen))
    await asyncio.sleep(0.01)
    assert not pump.done()  # blocked: finish() not called yet

    await session.finish()
    update = await pump
    assert update.final == "done"
    await agen.aclose()


async def test_collect_transcript_returns_final_text():
    backend = FakeSTTBackend(transcript="the answer", partials=["the "])
    text = await collect_transcript(backend, b"\x00\x00", 16000)
    assert text == "the answer"
    assert backend.sessions[0].fed == [(b"\x00\x00", 16000)]
```

(`FakeSTTBackend` is already imported at the top of the file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/backends/test_fake.py -v`
Expected: FAIL — `ImportError: cannot import name 'STTUpdate'`

- [ ] **Step 3: Implement protocol and fake session**

In `src/wyoming_mlx/backends/base.py`, replace the `STTBackend` protocol with (keep `TTSBackend` untouched; add `dataclass` import):

```python
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class STTUpdate:
    """Incremental transcription update.

    `text` is newly confirmed text (never retracted by later updates).
    `final` is the full transcript, set only on the last update of a session.
    """

    text: str = ""
    final: str | None = None


@runtime_checkable
class STTSession(Protocol):
    """One utterance's streaming transcription session."""

    async def feed(self, audio: bytes, sample_rate: int) -> None: ...
    async def finish(self) -> None: ...
    async def close(self) -> None: ...
    def updates(self) -> AsyncIterator[STTUpdate]: ...


@runtime_checkable
class STTBackend(Protocol):
    """Speech-to-text backend.

    Implementations must support multiple concurrent sessions; a real
    implementation may serialise GPU work internally.
    """

    async def transcribe(self, audio: bytes, sample_rate: int) -> str: ...

    def start_session(self) -> STTSession: ...


async def collect_transcript(backend: STTBackend, audio: bytes, sample_rate: int) -> str:
    """Run one complete utterance through a session and return the final text."""
    session = backend.start_session()
    try:
        await session.feed(audio, sample_rate)
        await session.finish()
        final = ""
        async for update in session.updates():
            if update.final is not None:
                final = update.final
        return final
    finally:
        await session.close()
```

(Note: `transcribe` stays on the protocol until Task 3 removes it — the HTTP route still calls it.)

In `src/wyoming_mlx/backends/fake.py`, replace `FakeSTTBackend` with:

```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from wyoming_mlx.backends.base import STTUpdate


class FakeSTTSession:
    """Scripted session: yields canned partials, then the final after finish()."""

    def __init__(self, partials: list[str], final: str) -> None:
        self._partials = partials
        self._final = final
        self.fed: list[tuple[bytes, int]] = []
        self.closed = False
        self._finished = asyncio.Event()

    async def feed(self, audio: bytes, sample_rate: int) -> None:
        self.fed.append((audio, sample_rate))

    async def finish(self) -> None:
        self._finished.set()

    async def close(self) -> None:
        self.closed = True
        self._finished.set()

    async def updates(self) -> AsyncIterator[STTUpdate]:
        for partial in self._partials:
            yield STTUpdate(text=partial)
        await self._finished.wait()
        yield STTUpdate(final=self._final)


class FakeSTTBackend:
    """In-memory STT backend for unit tests.

    Streams canned partials and a canned final transcript; records sessions.
    """

    def __init__(self, transcript: str = "", partials: list[str] | None = None) -> None:
        self.transcript = transcript
        self.partials = partials or []
        self.sessions: list[FakeSTTSession] = []
        self.calls: list[tuple[bytes, int]] = []

    async def transcribe(self, audio: bytes, sample_rate: int) -> str:
        self.calls.append((audio, sample_rate))
        return self.transcript

    def start_session(self) -> FakeSTTSession:
        session = FakeSTTSession(list(self.partials), self.transcript)
        self.sessions.append(session)
        return session
```

(`FakeTTSBackend` stays as is; keep its `AsyncIterator` import — it is now shared.)

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest tests/unit -v`
Expected: ALL PASS (old `transcribe`-based tests still green, new session tests green)

- [ ] **Step 5: Commit**

```bash
git add src/wyoming_mlx/backends/base.py src/wyoming_mlx/backends/fake.py tests/unit/backends/test_fake.py
git commit -m "feat: session-based streaming STT protocol with fake implementation"
```

---

### Task 2: Streaming Wyoming STT handler

**Files:**
- Modify: `src/wyoming_mlx/wyoming_servers/stt.py` (full rewrite)
- Test: `tests/unit/wyoming_servers/test_stt.py` (full rewrite)

- [ ] **Step 1: Rewrite the handler tests**

Replace the entire contents of `tests/unit/wyoming_servers/test_stt.py` with:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock

from wyoming.asr import Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info

from wyoming_mlx.backends.fake import FakeSTTBackend
from wyoming_mlx.wyoming_servers.stt import SttEventHandler


def _make_handler(backend: FakeSTTBackend, **kwargs) -> tuple[SttEventHandler, list[Event]]:
    handler = SttEventHandler(
        reader=MagicMock(spec=asyncio.StreamReader),
        writer=MagicMock(spec=asyncio.StreamWriter),
        backend=backend,
        info=Info(),
        **kwargs,
    )
    events: list[Event] = []
    handler.write_event = AsyncMock(side_effect=lambda e: events.append(e))
    return handler, events


def _pcm_chunk(n_samples: int = 1600) -> bytes:
    return b"\x00\x00" * n_samples


async def test_streams_chunks_then_final_transcript():
    backend = FakeSTTBackend(transcript="hello world", partials=["hello ", "world"])
    handler, events = _make_handler(backend)

    assert await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    assert await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=_pcm_chunk()).event()
    )
    assert await handler.handle_event(AudioStop().event())

    types = [e.type for e in events]
    assert types[0] == "transcript-start"
    chunks = [TranscriptChunk.from_event(e) for e in events if TranscriptChunk.is_type(e.type)]
    assert [c.text for c in chunks] == ["hello ", "world"]
    finals = [Transcript.from_event(e) for e in events if Transcript.is_type(e.type)]
    assert len(finals) == 1
    assert finals[0].text == "hello world"
    # final Transcript comes after all chunks, then transcript-stop ends the stream
    assert types.index("transcript") > types.index("transcript-chunk")
    assert types[-1] == "transcript-stop"


async def test_audio_is_fed_to_session_with_rate():
    backend = FakeSTTBackend(transcript="x")
    handler, _ = _make_handler(backend)

    await handler.handle_event(AudioStart(rate=24000, width=2, channels=1).event())
    await handler.handle_event(
        AudioChunk(rate=24000, width=2, channels=1, audio=b"\x01\x02").event()
    )
    await handler.handle_event(
        AudioChunk(rate=24000, width=2, channels=1, audio=b"\x03\x04").event()
    )
    await handler.handle_event(AudioStop().event())

    assert len(backend.sessions) == 1
    assert backend.sessions[0].fed == [(b"\x01\x02", 24000), (b"\x03\x04", 24000)]


async def test_chunk_without_audio_start_starts_session():
    backend = FakeSTTBackend(transcript="x")
    handler, events = _make_handler(backend)

    await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x01\x02").event()
    )
    await handler.handle_event(AudioStop().event())

    assert len(backend.sessions) == 1
    assert backend.sessions[0].fed == [(b"\x01\x02", 16000)]
    assert any(Transcript.is_type(e.type) for e in events)


async def test_audio_stop_without_start_is_ignored():
    backend = FakeSTTBackend(transcript="x")
    handler, events = _make_handler(backend)

    assert await handler.handle_event(AudioStop().event())

    assert backend.sessions == []
    assert events == []


async def test_responds_to_describe():
    backend = FakeSTTBackend(transcript="hello")
    handler, events = _make_handler(backend)

    assert await handler.handle_event(Describe().event())

    info_events = [e for e in events if Info.is_type(e.type)]
    assert len(info_events) == 1


async def test_rejects_overflowing_audio():
    backend = FakeSTTBackend(transcript="x")
    handler, events = _make_handler(backend, max_audio_bytes=10)

    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    assert await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x01\x02").event()
    )
    result = await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x03" * 10).event()
    )

    assert result is False
    assert backend.sessions[0].closed
    assert not any(Transcript.is_type(e.type) for e in events)


async def test_recovers_after_session_exception():
    backend = FakeSTTBackend(transcript="x")
    handler, events = _make_handler(backend)

    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    session = backend.sessions[0]
    session.finish = AsyncMock(side_effect=RuntimeError("GPU exploded"))

    await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x01\x02").event()
    )
    result = await handler.handle_event(AudioStop().event())

    assert result is True
    assert not any(Transcript.is_type(e.type) for e in events)

    # handler is reusable after the failure
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x05\x06").event()
    )
    assert await handler.handle_event(AudioStop().event())
    finals = [Transcript.from_event(e) for e in events if Transcript.is_type(e.type)]
    assert [f.text for f in finals] == ["x"]


async def test_disconnect_aborts_open_session():
    backend = FakeSTTBackend(transcript="x")
    handler, _ = _make_handler(backend)

    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.disconnect()

    assert backend.sessions[0].closed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/wyoming_servers/test_stt.py -v`
Expected: FAIL — `TypeError` (handler still buffers; no session use) and assertion errors

- [ ] **Step 3: Rewrite the handler**

Replace the entire contents of `src/wyoming_mlx/wyoming_servers/stt.py` with:

```python
from __future__ import annotations

import asyncio
import contextlib
import logging

from wyoming.asr import Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from wyoming_mlx.backends.base import STTBackend, STTSession

log = logging.getLogger(__name__)


class SttEventHandler(AsyncEventHandler):
    """Wyoming event handler for one STT client connection.

    Streams audio into a backend session as it arrives and emits
    transcript-start / transcript-chunk events for confirmed text, followed
    by the final transcript (for non-streaming clients) and transcript-stop.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        backend: STTBackend,
        info: Info,
        max_audio_bytes: int = 100_000_000,
    ) -> None:
        super().__init__(reader=reader, writer=writer)
        self._backend = backend
        self._info = info
        self._max_audio_bytes = max_audio_bytes
        self._session: STTSession | None = None
        self._pump_task: asyncio.Task[str] | None = None
        self._bytes_fed = 0

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True

        if AudioStart.is_type(event.type):
            await self._abort_session()
            self._start_session()
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if self._session is None:
                self._start_session()
            assert self._session is not None
            self._bytes_fed += len(chunk.audio)
            if self._bytes_fed > self._max_audio_bytes:
                log.warning("STT audio exceeded limit of %d bytes", self._max_audio_bytes)
                await self._abort_session()
                return False
            try:
                await self._session.feed(chunk.audio, chunk.rate)
            except Exception:
                log.exception("feeding audio to STT session failed")
                await self._abort_session()
            return True

        if AudioStop.is_type(event.type):
            if self._session is None:
                log.warning("audio-stop received without audio; ignoring")
                return True
            session = self._session
            pump_task = self._pump_task
            self._session = None
            self._pump_task = None
            self._bytes_fed = 0
            try:
                await session.finish()
                final = await pump_task if pump_task is not None else ""
            except Exception:
                log.exception("transcription failed")
                if pump_task is not None:
                    pump_task.cancel()
                await self._close_quietly(session)
                return True
            await self.write_event(Transcript(text=final).event())
            await self.write_event(TranscriptStop().event())
            await self._close_quietly(session)
            return True

        return True

    async def disconnect(self) -> None:
        await self._abort_session()

    def _start_session(self) -> None:
        self._session = self._backend.start_session()
        self._bytes_fed = 0
        self._pump_task = asyncio.create_task(self._pump(self._session))

    async def _pump(self, session: STTSession) -> str:
        """Emit streaming events for session updates; return the final text."""
        await self.write_event(TranscriptStart().event())
        final = ""
        async for update in session.updates():
            if update.text:
                await self.write_event(TranscriptChunk(text=update.text).event())
            if update.final is not None:
                final = update.final
        return final

    async def _abort_session(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump_task
            self._pump_task = None
        if self._session is not None:
            await self._close_quietly(self._session)
            self._session = None
        self._bytes_fed = 0

    @staticmethod
    async def _close_quietly(session: STTSession) -> None:
        with contextlib.suppress(Exception):
            await session.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/wyoming_servers/test_stt.py -v`
Expected: ALL PASS

Note: `contextlib.suppress(asyncio.CancelledError, Exception)` — CancelledError is not an
Exception subclass in 3.12, both are needed. If ruff complains about suppressing broad
Exception (`BLE`/`S110` are not in the selected rule set, so it shouldn't), do not "fix" it
by narrowing; abort paths must not raise.

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest tests/unit -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/wyoming_mlx/wyoming_servers/stt.py tests/unit/wyoming_servers/test_stt.py
git commit -m "feat: stream transcript-start/chunk/stop events from Wyoming STT handler"
```

---

### Task 3: HTTP route via collect_transcript; drop legacy transcribe()

**Files:**
- Modify: `src/wyoming_mlx/http/routes.py:118-130`
- Modify: `src/wyoming_mlx/backends/base.py` (remove `transcribe` from protocol)
- Modify: `src/wyoming_mlx/backends/fake.py` (remove `transcribe` and `calls`)
- Test: `tests/unit/http/test_routes.py`, `tests/unit/backends/test_fake.py`

- [ ] **Step 1: Update the transcription route**

In `src/wyoming_mlx/http/routes.py`, change the import:

```python
from wyoming_mlx.backends.base import STTBackend, TTSBackend, collect_transcript
```

and in the `transcribe` route handler replace

```python
        text = await stt.transcribe(pcm, rate)
```

with

```python
        text = await collect_transcript(stt, pcm, rate)
```

- [ ] **Step 2: Remove the legacy method**

- In `src/wyoming_mlx/backends/base.py`: delete the line
  `async def transcribe(self, audio: bytes, sample_rate: int) -> str: ...` from `STTBackend`.
- In `src/wyoming_mlx/backends/fake.py`: delete `FakeSTTBackend.transcribe` and the
  `self.calls` attribute.

- [ ] **Step 3: Fix any tests that referenced the legacy API**

Run: `uv run pytest tests/unit -v`

For failures referencing `.transcribe` or `.calls` on the fake (expected in
`tests/unit/http/test_routes.py` and possibly `tests/unit/backends/test_fake.py`):
- assertions like `backend.calls == [...]` become `backend.sessions[0].fed == [...]`
- assertions counting calls become `len(backend.sessions) == 1`

The HTTP route's response contract (`{"text": ...}`) is unchanged, so route tests should
only need those substitutions, not behavioural changes.

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest tests/unit -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/wyoming_mlx/http/routes.py src/wyoming_mlx/backends/base.py src/wyoming_mlx/backends/fake.py tests/unit
git commit -m "feat: HTTP transcription endpoint uses streaming sessions; drop batch transcribe()"
```

---

### Task 4: WhisperLiveKit backend (replaces mlx_stt.py)

**Files:**
- Create: `src/wyoming_mlx/backends/wlk_stt.py`
- Delete: `src/wyoming_mlx/backends/mlx_stt.py`
- Create: `tests/unit/backends/test_wlk_stt.py`
- Delete: `tests/unit/backends/test_mlx_stt.py`

- [ ] **Step 1: Write failing tests for the pure helpers**

Create `tests/unit/backends/test_wlk_stt.py`:

```python
"""Unit tests for WhisperLiveKit backend helpers (no whisperlivekit install required)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from wyoming_mlx.backends.wlk_stt import (
    WHISPER_SAMPLE_RATE,
    _delta,
    _joined_text,
    _resample_to_16k,
    resample_pcm16,
)


def test_resample_passthrough_at_16k():
    pcm = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    out = _resample_to_16k(pcm, 16000)
    assert out is pcm or np.array_equal(out, pcm)
    assert out.dtype == np.float32


def test_resample_24k_to_16k_changes_length_correctly():
    pcm = np.zeros(24000, dtype=np.float32)
    out = _resample_to_16k(pcm, 24000)
    assert out.dtype == np.float32
    assert out.shape == (WHISPER_SAMPLE_RATE,)


def test_resample_preserves_sinusoid_frequency():
    sr_in = 24000
    duration = 0.5
    freq = 440.0
    t = np.arange(int(sr_in * duration), dtype=np.float32) / sr_in
    pcm = np.sin(2 * np.pi * freq * t).astype(np.float32)
    out = _resample_to_16k(pcm, sr_in)
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(len(out), d=1 / 16000)[int(np.argmax(spectrum))]
    assert abs(peak_hz - freq) < 5


def test_resample_pcm16_passthrough_at_16k_is_lossless():
    audio = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16).tobytes()
    assert resample_pcm16(audio, 16000) == audio


def test_resample_pcm16_halves_48k_input():
    audio = np.zeros(4800, dtype=np.int16).tobytes()
    out = resample_pcm16(audio, 48000)
    assert len(out) == 1600 * 2  # 1600 samples of int16


def test_joined_text_skips_empty_segments():
    lines = [
        SimpleNamespace(text="hello"),
        SimpleNamespace(text=None),
        SimpleNamespace(text="  "),
        SimpleNamespace(text="world"),
    ]
    assert _joined_text(lines) == "hello world"


def test_delta_returns_new_suffix():
    assert _delta("hello ", "hello world") == "world"


def test_delta_empty_when_not_a_prefix():
    assert _delta("hello there", "hello world") == ""


def test_delta_from_empty_emits_everything():
    assert _delta("", "hello") == "hello"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/backends/test_wlk_stt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wyoming_mlx.backends.wlk_stt'`

- [ ] **Step 3: Create the backend**

Create `src/wyoming_mlx/backends/wlk_stt.py`:

```python
"""WhisperLiveKit streaming STT backend (AlignAtt / SimulStreaming on MLX)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from math import gcd
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from wyoming_mlx.backends.base import STTUpdate

try:
    from whisperlivekit import (  # pyright: ignore[reportMissingImports]
        AudioProcessor as _AudioProcessor,
    )
    from whisperlivekit import (  # pyright: ignore[reportMissingImports]
        TranscriptionEngine as _TranscriptionEngine,
    )

    _WLK_AVAILABLE = True
except ImportError:
    _AudioProcessor = None  # type: ignore[assignment]
    _TranscriptionEngine = None  # type: ignore[assignment]
    _WLK_AVAILABLE = False

log = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


def _resample_to_16k(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample float32 mono audio to Whisper's required 16 kHz.

    Whisper produces empty/garbled transcripts when fed audio at the wrong
    rate, since its mel spectrogram is computed assuming 16 kHz input.
    """
    if sample_rate == WHISPER_SAMPLE_RATE:
        return pcm
    g = gcd(sample_rate, WHISPER_SAMPLE_RATE)
    up = WHISPER_SAMPLE_RATE // g
    down = sample_rate // g
    return resample_poly(pcm, up, down).astype(np.float32)


def resample_pcm16(audio: bytes, sample_rate: int) -> bytes:
    """Resample int16 mono PCM bytes to 16 kHz int16 mono PCM bytes."""
    if sample_rate == WHISPER_SAMPLE_RATE:
        return audio
    pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    pcm16k = _resample_to_16k(pcm, sample_rate)
    return (np.clip(pcm16k, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _joined_text(lines: list[Any]) -> str:
    """Join WLK confirmed segments into a single transcript string."""
    return " ".join(seg.text.strip() for seg in lines if seg.text and seg.text.strip())


def _delta(emitted: str, confirmed: str) -> str:
    """Newly confirmed suffix, or "" if `confirmed` is not an extension of `emitted`.

    WLK confirmed lines only grow, so the prefix property normally holds; if a
    text revision ever breaks it we emit nothing and let the final transcript
    (which is authoritative) carry the correction.
    """
    if confirmed.startswith(emitted):
        return confirmed[len(emitted) :]
    return ""


class _WLKSession:
    """One utterance: wraps a WhisperLiveKit AudioProcessor."""

    def __init__(self, engine: Any) -> None:
        assert _AudioProcessor is not None
        self._processor = _AudioProcessor(transcription_engine=engine)
        self._results: AsyncIterator[Any] | None = None
        self._emitted = ""

    async def _ensure_started(self) -> None:
        if self._results is None:
            self._results = await self._processor.create_tasks()

    async def feed(self, audio: bytes, sample_rate: int) -> None:
        await self._ensure_started()
        await self._processor.process_audio(resample_pcm16(audio, sample_rate))

    async def finish(self) -> None:
        await self._ensure_started()
        # Empty message triggers WLK's flush + stop sequence; the results
        # generator terminates once all processing tasks finish.
        await self._processor.process_audio(None)

    async def close(self) -> None:
        await self._processor.cleanup()

    async def updates(self) -> AsyncIterator[STTUpdate]:
        await self._ensure_started()
        assert self._results is not None
        confirmed = ""
        buffer_tail = ""
        async for front in self._results:
            if front.status == "error":
                raise RuntimeError(f"WhisperLiveKit error: {front.error}")
            confirmed = _joined_text(front.lines)
            buffer_tail = front.buffer_transcription or ""
            new_text = _delta(self._emitted, confirmed)
            if new_text:
                self._emitted = confirmed
                yield STTUpdate(text=new_text)
        # Anything still unconfirmed when the stream ends belongs in the final.
        final = f"{confirmed} {buffer_tail.strip()}".strip() if buffer_tail.strip() else confirmed
        yield STTUpdate(final=final)


class WhisperLiveKitBackend:
    """STT backend powered by WhisperLiveKit (SimulStreaming policy, MLX).

    The TranscriptionEngine is a process-wide singleton holding the model;
    it is created eagerly so startup fails fast if the model can't load.
    Each session gets its own AudioProcessor and may run concurrently.
    """

    def __init__(self, model: str = "large-v3-turbo") -> None:
        assert _WLK_AVAILABLE, "whisperlivekit required"
        assert _TranscriptionEngine is not None
        log.info("Loading WhisperLiveKit engine (model=%s) …", model)
        self._engine = _TranscriptionEngine(
            model_size=model,
            backend="mlx-whisper",
            backend_policy="simulstreaming",
            pcm_input=True,
            log_level="WARNING",
        )
        log.info("[STT] WhisperLiveKit engine ready")

    def start_session(self) -> _WLKSession:
        return _WLKSession(self._engine)
```

Delete the old backend and its tests:

```bash
git rm src/wyoming_mlx/backends/mlx_stt.py tests/unit/backends/test_mlx_stt.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit -v`
Expected: ALL PASS (helper tests run without whisperlivekit installed; `_WLKSession` /
`WhisperLiveKitBackend` are only exercised by integration tests)

- [ ] **Step 5: Commit**

```bash
git add src/wyoming_mlx/backends/wlk_stt.py tests/unit/backends/test_wlk_stt.py
git commit -m "feat: WhisperLiveKit streaming backend replaces mlx_whisper batch backend"
```

---

### Task 5: Dependencies, config default, main wiring

**Files:**
- Modify: `pyproject.toml:17-31`
- Modify: `src/wyoming_mlx/config.py:32`
- Modify: `src/wyoming_mlx/__main__.py` (imports, `_ensure_models_cached`, `_build_stt_info`, `main`)
- Test: `tests/unit/test_config.py` (only if it asserts the old default)

- [ ] **Step 1: Swap dependencies**

In `pyproject.toml` `dependencies`, replace

```toml
    "mlx-whisper>=0.4.3; sys_platform == 'darwin'",
```

with

```toml
    "whisperlivekit[mlx-whisper]==0.2.22; sys_platform == 'darwin'",
```

(Exact pin: WLK is 0.x and its internal API — `FrontData`, `AudioProcessor` — is what we
program against. Leave the `mlx` dependency line alone.)

Run: `uv sync` — expected to resolve and install whisperlivekit on this Mac.

- [ ] **Step 2: Change the model default**

In `src/wyoming_mlx/config.py`:

```python
class ModelsConfig(BaseModel):
    whisper: str = "large-v3-turbo"
```

(The value is now a WhisperLiveKit model name, e.g. `tiny`/`base`/`small`/`medium`/
`large-v3`/`large-v3-turbo` — WLK maps it to an MLX checkpoint itself.)

Run: `uv run pytest tests/unit/test_config.py -v` — if a test asserts the old
`mlx-community/distil-whisper-large-v3` default, update it to `large-v3-turbo`.

- [ ] **Step 3: Update `__main__.py`**

Four changes:

1. `_ensure_models_cached`: remove the `whisper_model_id` parameter and the entire
   "Whisper model" `try` block (the engine downloads its model during eager init);
   `cache_key = kokoro_model_id`. Update the call site to
   `_ensure_models_cached(cfg.models.kokoro)`.

2. `_build_stt_info`: advertise streaming and the new engine:

```python
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
```

3. The lazy backend import in `main()`:

```python
    try:
        from wyoming_mlx.backends.mlx_tts import KokoroBackend
        from wyoming_mlx.backends.wlk_stt import WhisperLiveKitBackend
    except ImportError:
        log.error(
            "MLX backends not available (mlx, whisperlivekit, kokoro packages missing). "
            "Install the MLX dependencies to enable real models."
        )
        sys.exit(1)
```

4. Backend construction:

```python
    stt_backend = WhisperLiveKitBackend(model=cfg.models.whisper)
```

- [ ] **Step 4: Run the full unit suite and type check**

Run: `uv run pytest tests/unit -v && uv run pyright`
Expected: ALL PASS, no new pyright errors. (`tests/unit/test_main.py` uses fakes and
should pass unmodified.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/wyoming_mlx/config.py src/wyoming_mlx/__main__.py tests/unit/test_config.py
git commit -m "feat: wire WhisperLiveKit backend into startup; advertise transcript streaming"
```

---

### Task 6: Integration test and README

**Files:**
- Modify: `tests/integration/test_roundtrip.py`
- Modify: `README.md` (lines mentioning distil-whisper / mlx-whisper: ~3, ~27, ~74)

- [ ] **Step 1: Update the roundtrip integration test**

Replace the contents of `tests/integration/test_roundtrip.py` with:

```python
"""Integration test: TTS → streaming STT round-trip with real MLX models."""

import pytest

from wyoming_mlx.backends.mlx_tts import KokoroBackend


@pytest.mark.integration
async def test_tts_streaming_stt_roundtrip():
    """Kokoro synthesises → WhisperLiveKit streams a transcript back.

    Asserts both correctness (word overlap with the input) and streaming
    behaviour (at least one confirmed partial arrives before the final).
    """
    pytest.importorskip("whisperlivekit")
    pytest.importorskip("kokoro")

    from wyoming_mlx.backends.wlk_stt import WhisperLiveKitBackend

    tts = KokoroBackend(model_id="hexgrad/Kokoro-82M", voice="af_heart")
    stt = WhisperLiveKitBackend()

    text = "The quick brown fox jumps over the lazy dog"

    audio_chunks = []
    async for chunk in tts.synthesize(text, "af_heart"):
        audio_chunks.append(chunk)
    audio = b"".join(audio_chunks)
    assert len(audio) > 0, "TTS produced no audio output"

    session = stt.start_session()
    # Feed in ~0.5 s chunks to mimic a live stream.
    bytes_per_half_sec = tts.sample_rate  # sample_rate samples/s * 2 bytes / 2
    for i in range(0, len(audio), bytes_per_half_sec):
        await session.feed(audio[i : i + bytes_per_half_sec], tts.sample_rate)
    await session.finish()

    partials: list[str] = []
    final = ""
    async for update in session.updates():
        if update.text:
            partials.append(update.text)
        if update.final is not None:
            final = update.final
    await session.close()

    assert final, "STT produced empty transcript"
    assert partials, "no streaming partials were emitted before the final transcript"

    original_words = set(text.lower().split())
    transcribed_words = set(final.lower().replace(".", "").replace(",", "").split())
    overlap = original_words & transcribed_words
    ratio = len(overlap) / len(original_words)
    assert ratio >= 0.5, (
        f"Word overlap too low: {ratio:.0%} — original={original_words}, transcript='{final}'"
    )
```

- [ ] **Step 2: Update README**

Adjust the three STT references found via `grep -n "distil\|mlx-whisper" README.md`:
- line ~3: "STT (distil-whisper)" → "streaming STT (Whisper via WhisperLiveKit)"
- line ~27: "speech-to-text runs Whisper on Metal via `mlx-whisper`" → "speech-to-text
  streams through WhisperLiveKit's SimulStreaming (AlignAtt) policy on MLX, so partial
  transcripts arrive while you're still speaking"
- line ~74: "distil-whisper-large-v3 (MLX)" → "whisper-large-v3-turbo (MLX, streaming)"

Mention streaming support in whatever section describes Home Assistant integration:
streaming transcripts require HA 2025.7+ (older versions still get the final transcript).

- [ ] **Step 3: Run ruff and the unit suite**

Run: `uv run ruff check . && uv run pytest tests/unit -q`
Expected: clean, ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_roundtrip.py README.md
git commit -m "test+docs: streaming STT integration roundtrip and README update"
```

---

### Task 7: Full verification on Apple Silicon

**Files:** none (verification only)

- [ ] **Step 1: Full local gate**

Run: `uv run ruff check . && uv run pyright && uv run pytest tests/unit -q`
Expected: all clean. Fix anything that isn't before proceeding.

- [ ] **Step 2: Run the integration suite (downloads whisper-large-v3-turbo ~1.6 GB on first run)**

Run: `uv run pytest --integration tests/integration -v`
Expected: PASS, including the `partials` assertion (proves streaming actually streams).

If the partials assertion fails (VAD swallowing the short utterance), diagnose before
weakening the test — likely knobs: `vac=False` or `audio_min_len` in
`WhisperLiveKitBackend.__init__` kwargs. Any such change is a deliberate decision to
record in the spec's config section, not a silent tweak.

- [ ] **Step 3: Smoke the real server**

Run: `uv run wyoming-mlx --log-level DEBUG` (Ctrl-C after startup) — expected: engine
loads eagerly, `[STT] WhisperLiveKit engine ready` appears, all three servers bind.

- [ ] **Step 4: Commit any fixes; do not commit if everything already passed**

---

### Task 8: PR

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin feat/streaming-stt-whisperlivekit
gh pr create --title "feat: streaming speech-to-text via WhisperLiveKit" --body "$(cat <<'EOF'
## Summary
- Replace the batch mlx_whisper STT backend with WhisperLiveKit (AlignAtt SimulStreaming on MLX)
- Wyoming STT now streams transcript-start/transcript-chunk events while audio arrives (HA 2025.7+), with the final Transcript kept for older clients
- HTTP /v1/audio/transcriptions reuses the same engine via one-shot sessions
- Default model: large-v3-turbo (distil checkpoints lack AlignAtt alignment heads)

Design spec: docs/superpowers/specs/2026-06-12-whisperlivekit-streaming-stt-design.md

## Test plan
- [ ] Unit suite (Linux-safe, fake backend) — CI
- [ ] `uv run pytest --integration tests/integration -v` on Apple Silicon
- [ ] Manual: HA voice pipeline against the Mac Studio

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** protocol (Task 1), handler + streaming events + Info flag (Tasks 2, 5),
  HTTP reuse (Task 3), WLK backend with eager init + confirmed-only chunks (Task 4),
  config/deps (Task 5), testing incl. short-utterance validation (Tasks 6–7). Error
  handling: feed/finish failures and oversize utterances in Task 2; engine-load failure
  surfaces via eager init in `main()` (Task 5).
- **Known judgment call encoded:** `_delta` yields nothing on prefix violation; final
  transcript is authoritative.
- **Out of scope (zag later):** the now-possibly-redundant direct `mlx` dependency;
  TODO.md's eager-load item for TTS (STT half is resolved by this work).
