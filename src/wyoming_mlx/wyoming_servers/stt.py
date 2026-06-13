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
                await self._terminate_stream()
                return False
            try:
                await self._session.feed(chunk.audio, chunk.rate)
            except Exception:
                log.exception("feeding audio to STT session failed")
                await self._abort_session()
                await self._terminate_stream()
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
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await pump_task
                await self._close_quietly(session)
                await self._terminate_stream()
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

    async def _terminate_stream(self) -> None:
        """Emit an empty final transcript + stop so a client that already saw
        transcript-start isn't left waiting after an aborted utterance."""
        await self.write_event(Transcript(text="").event())
        await self.write_event(TranscriptStop().event())

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
