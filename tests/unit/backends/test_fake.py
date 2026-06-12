import asyncio

import pytest

from wyoming_mlx.backends.base import STTBackend, STTUpdate, TTSBackend, collect_transcript
from wyoming_mlx.backends.fake import FakeSTTBackend, FakeTTSBackend


def test_fakes_satisfy_protocols():
    assert isinstance(FakeSTTBackend(), STTBackend)
    assert isinstance(FakeTTSBackend(chunks=[b""]), TTSBackend)


@pytest.mark.asyncio
async def test_fake_stt_records_calls():
    backend = FakeSTTBackend(transcript="x")
    await collect_transcript(backend, b"abc", sample_rate=16000)
    await collect_transcript(backend, b"def", sample_rate=22050)
    assert len(backend.sessions) == 2
    assert backend.sessions[0].fed == [(b"abc", 16000)]
    assert backend.sessions[1].fed == [(b"def", 22050)]


@pytest.mark.asyncio
async def test_fake_tts_yields_canned_chunks():
    backend = FakeTTSBackend(chunks=[b"AAAA", b"BBBB"], voices=["v1", "v2"])
    out = [chunk async for chunk in backend.synthesize("hi", voice="v1")]
    assert out == [b"AAAA", b"BBBB"]


@pytest.mark.asyncio
async def test_fake_tts_voices_attribute():
    backend = FakeTTSBackend(chunks=[b""], voices=["alice", "bob"])
    assert backend.voices == ["alice", "bob"]


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
    pump = asyncio.create_task(anext(agen))
    await asyncio.sleep(0)
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
