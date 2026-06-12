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
