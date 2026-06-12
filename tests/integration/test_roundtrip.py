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
    bytes_per_half_sec = tts.sample_rate  # sample_rate samples/s * 2 bytes/sample / 2
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
