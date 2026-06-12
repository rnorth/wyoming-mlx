"""Integration test: TTS → STT round-trip with real MLX models."""

import pytest

from wyoming_mlx.backends.base import collect_transcript
from wyoming_mlx.backends.mlx_tts import KokoroBackend
from wyoming_mlx.backends.wlk_stt import WhisperLiveKitBackend


@pytest.mark.integration
async def test_tts_stt_roundtrip():
    """Kokoro synthesises → WhisperLiveKit transcribes → transcript shares words with input."""
    # Skip unless --integration flag is passed
    pytest.importorskip("whisperlivekit")
    pytest.importorskip("kokoro")

    tts = KokoroBackend(model_id="hexgrad/Kokoro-82M", voice="af_heart")
    stt = WhisperLiveKitBackend()

    text = "The quick brown fox jumps over the lazy dog"

    # Synthesise
    audio_chunks = []
    async for chunk in tts.synthesize(text, "af_heart"):
        audio_chunks.append(chunk)
    audio = b"".join(audio_chunks)
    assert len(audio) > 0, "TTS produced no audio output"

    # Transcribe
    transcript = await collect_transcript(stt, audio, tts.sample_rate)
    assert transcript, "STT produced empty transcript"

    # Check word overlap (at least 50%)
    original_words = set(text.lower().split())
    transcribed_words = set(transcript.lower().split())
    overlap = original_words & transcribed_words
    ratio = len(overlap) / len(original_words) if original_words else 0
    assert ratio >= 0.5, (
        f"Word overlap too low: {ratio:.0%} — "
        f"original={original_words}, transcript='{transcript.strip()}'"
    )
