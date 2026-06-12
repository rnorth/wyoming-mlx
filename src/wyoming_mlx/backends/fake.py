from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator

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

    async def updates(self) -> AsyncGenerator[STTUpdate, None]:
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

    def start_session(self) -> FakeSTTSession:
        session = FakeSTTSession(list(self.partials), self.transcript)
        self.sessions.append(session)
        return session


class FakeTTSBackend:
    """In-memory TTS backend for unit tests.

    Yields a fixed list of chunks for every synthesize call.
    """

    def __init__(
        self,
        chunks: list[bytes],
        voices: list[str] | None = None,
        sample_rate: int = 24000,
    ) -> None:
        self._chunks = list(chunks)
        self.voices = voices or ["default"]
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, str]] = []

    async def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]:
        self.calls.append((text, voice))
        for chunk in self._chunks:
            yield chunk
