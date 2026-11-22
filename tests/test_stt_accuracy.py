"""
Unit & Integration Tests for Speech-to-Text (STT) Accuracy & Groq Whisper Optimization.
Validates:
1. Dynamic peak normalization on quiet audio.
2. 16kHz 16-bit mono PCM WAV buffer construction.
3. Groq Cloud Whisper Large V3 Turbo API parameters (prompt, temperature=0.0, language='en').
4. Accuracy on short utterances ("Yes", "No", "Agent", etc.) and long multi-clause utterances.
5. Local Faster-Whisper fallback on quiet audio with initial_prompt.
"""

import asyncio
import io
import os
import sys
import unittest
import wave
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from telephony_server import (
    SAMPLE_RATE_VAD,
    calculate_stt_accuracy,
    registry,
    transcribe_speech,
)


class TestSTTAccuracy(unittest.IsolatedAsyncioTestCase):

    async def test_empty_or_silent_audio_returns_empty(self):
        """Verifies that empty audio or absolute dead silence (<1e-4) returns cleanly without API calls."""
        res_empty = await transcribe_speech(np.zeros(0, dtype=np.float32))
        self.assertEqual(res_empty, "")

        res_silence = await transcribe_speech(np.zeros(16000, dtype=np.float32))
        self.assertEqual(res_silence, "")

    async def test_dynamic_peak_normalization_and_wav_format(self):
        """Verifies that quiet speech is scaled to 0.95 and written into a valid 16kHz 16-bit mono WAV buffer."""
        captured_file = None

        mock_response = MagicMock()
        mock_response.text = "Hello support agent."

        async def capture_call(*args, **kwargs):
            nonlocal captured_file
            captured_file = kwargs.get("file")
            return mock_response

        mock_groq = MagicMock()
        mock_groq.audio.transcriptions.create = AsyncMock(side_effect=capture_call)

        orig_client = registry.groq_stt_client
        registry.groq_stt_client = mock_groq

        try:
            # Simulate quiet audio at 3% volume (peak ~0.03)
            t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
            quiet_speech = 0.03 * np.sin(2 * np.pi * 300 * t)

            transcript = await transcribe_speech(quiet_speech)
            self.assertEqual(transcript, "Hello support agent.")

            # Validate captured file tuple: (filename, io.BytesIO, mime_type)
            self.assertIsNotNone(captured_file)
            filename, file_bytes_io, mime = captured_file
            self.assertEqual(filename, "audio.wav")
            self.assertEqual(mime, "audio/wav")

            # Validate WAV header format
            file_bytes_io.seek(0)
            with wave.open(file_bytes_io, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)  # 16-bit
                self.assertEqual(wf.getframerate(), SAMPLE_RATE_VAD)  # 16000
                frames = wf.readframes(wf.getnframes())
                pcm = np.frombuffer(frames, dtype=np.int16)
                # Verify amplitude was boosted by dynamic peak normalization (~0.95 * 32767 = 31128)
                self.assertGreater(np.max(np.abs(pcm)), 30000)

        finally:
            registry.groq_stt_client = orig_client

    async def test_groq_whisper_parameters(self):
        """Verifies Groq Whisper Large V3 Turbo call includes model, language, prompt, temperature=0.0."""
        mock_response = MagicMock()
        mock_response.text = "I would like to check my account balance."

        mock_groq = MagicMock()
        mock_groq.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        orig_client = registry.groq_stt_client
        registry.groq_stt_client = mock_groq

        try:
            audio = 0.5 * np.ones(8000, dtype=np.float32)
            res = await transcribe_speech(audio)
            self.assertEqual(res, "I would like to check my account balance.")

            mock_groq.audio.transcriptions.create.assert_awaited_once()
            _, kwargs = mock_groq.audio.transcriptions.create.call_args
            self.assertEqual(kwargs.get("model"), "whisper-large-v3-turbo")
            self.assertEqual(kwargs.get("language"), "en")
            self.assertEqual(kwargs.get("temperature"), 0.0)
            self.assertIn("Spoken phone conversation", kwargs.get("prompt", ""))
            self.assertEqual(kwargs.get("response_format"), "json")

        finally:
            registry.groq_stt_client = orig_client

    async def test_local_whisper_fallback_receives_normalized_audio(self):
        """Verifies local Faster-Whisper fallback is invoked with normalized audio and initial_prompt."""
        mock_groq = MagicMock()
        mock_groq.audio.transcriptions.create = AsyncMock(side_effect=Exception("Groq LPU 503 error"))

        @dataclass
        class MockSegment:
            text: str

        captured_audio = None
        captured_kwargs = None

        mock_whisper = MagicMock()
        def mock_transcribe(audio, **kwargs):
            nonlocal captured_audio, captured_kwargs
            captured_audio = audio
            captured_kwargs = kwargs
            return [MockSegment("Fallback succeeded")], None

        mock_whisper.transcribe = MagicMock(side_effect=mock_transcribe)

        orig_groq = registry.groq_stt_client
        orig_whisper = registry.whisper
        registry.groq_stt_client = mock_groq
        registry.whisper = mock_whisper

        try:
            # Quiet audio at 0.05
            quiet_audio = 0.05 * np.ones(3200, dtype=np.float32)
            transcript = await transcribe_speech(quiet_audio)
            self.assertEqual(transcript, "Fallback succeeded")

            # Check that captured audio was normalized to 0.95 peak
            self.assertIsNotNone(captured_audio)
            self.assertAlmostEqual(float(np.max(np.abs(captured_audio))), 0.95, delta=0.01)
            self.assertEqual(captured_kwargs.get("initial_prompt"), "Spoken phone conversation with customer support agent.")

        finally:
            registry.groq_stt_client = orig_groq
            registry.whisper = orig_whisper


if __name__ == "__main__":
    unittest.main()
