import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from wyoming.asr import Transcript, TranscriptChunk, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info

from wyoming_mlx.backends.fake import FakeSTTBackend
from wyoming_mlx.wyoming_servers.stt import SttEventHandler


def _make_handler(backend: FakeSTTBackend, **kwargs: Any) -> tuple[SttEventHandler, list[Event]]:
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
    # transcript-start first, ALL chunks before the final Transcript, stop last
    chunk_indices = [i for i, t in enumerate(types) if t == "transcript-chunk"]
    transcript_idx = types.index("transcript")
    assert all(i < transcript_idx for i in chunk_indices)
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
    # overflow still terminates the stream cleanly before disconnecting
    finals = [Transcript.from_event(e) for e in events if Transcript.is_type(e.type)]
    assert [f.text for f in finals] == [""]
    assert events[-1].type == "transcript-stop"


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
    # a failure still terminates the stream: empty final transcript + stop
    finals = [Transcript.from_event(e) for e in events if Transcript.is_type(e.type)]
    assert [f.text for f in finals] == [""]
    assert any(TranscriptStop.is_type(e.type) for e in events)
    assert events[-1].type == "transcript-stop"

    # handler is reusable after the failure
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x05\x06").event()
    )
    assert await handler.handle_event(AudioStop().event())
    finals = [Transcript.from_event(e) for e in events if Transcript.is_type(e.type)]
    assert [f.text for f in finals] == ["", "x"]


async def test_pump_failure_still_terminates_stream():
    backend = FakeSTTBackend(transcript="x")
    handler, events = _make_handler(backend)

    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    session = backend.sessions[0]

    async def boom():
        raise RuntimeError("decoder died")
        yield  # pragma: no cover - makes this an async generator

    session.updates = boom

    await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x01\x02").event()
    )
    result = await handler.handle_event(AudioStop().event())

    assert result is True
    finals = [Transcript.from_event(e) for e in events if Transcript.is_type(e.type)]
    assert [f.text for f in finals] == [""]
    assert events[-1].type == "transcript-stop"


async def test_feed_failure_terminates_stream():
    backend = FakeSTTBackend(transcript="x")
    handler, events = _make_handler(backend)

    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    session = backend.sessions[0]
    session.feed = AsyncMock(side_effect=RuntimeError("device gone"))

    # feed failure aborts the session but keeps the connection open …
    assert await handler.handle_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\x01\x02").event()
    )
    # … and still terminates the stream so the client isn't left waiting
    finals = [Transcript.from_event(e) for e in events if Transcript.is_type(e.type)]
    assert [f.text for f in finals] == [""]
    assert events[-1].type == "transcript-stop"

    # a subsequent audio-stop is a no-op (session already gone)
    assert await handler.handle_event(AudioStop().event())
    finals = [Transcript.from_event(e) for e in events if Transcript.is_type(e.type)]
    assert [f.text for f in finals] == [""]


async def test_two_successful_utterances_back_to_back():
    backend = FakeSTTBackend(transcript="hello", partials=["hel", "lo"])
    handler, events = _make_handler(backend)

    for _ in range(2):
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=_pcm_chunk()).event()
        )
        await handler.handle_event(AudioStop().event())

    assert len(backend.sessions) == 2
    finals = [Transcript.from_event(e) for e in events if Transcript.is_type(e.type)]
    assert [f.text for f in finals] == ["hello", "hello"]
    assert [e.type for e in events].count("transcript-start") == 2
    assert [e.type for e in events].count("transcript-stop") == 2


async def test_empty_utterance_start_then_stop():
    backend = FakeSTTBackend(transcript="")
    handler, events = _make_handler(backend)

    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    assert await handler.handle_event(AudioStop().event())

    assert len(backend.sessions) == 1
    finals = [Transcript.from_event(e) for e in events if Transcript.is_type(e.type)]
    assert [f.text for f in finals] == [""]
    assert events[-1].type == "transcript-stop"


async def test_disconnect_aborts_open_session():
    backend = FakeSTTBackend(transcript="x")
    handler, _ = _make_handler(backend)

    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.disconnect()

    assert backend.sessions[0].closed
