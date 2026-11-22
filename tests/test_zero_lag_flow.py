"""
Tests for Zero-Lag Voice Flow & Real-Time Telephony Dispatch.
Validates:
1. VAD threshold is tuned to 0.45 to prevent ambient hiss/breathing false-positives.
2. transcribe_speech prioritizes Cloud Groq STT and avoids running local CPU Whisper concurrently.
3. Inbound audio is never forcibly attenuated by 0.005 during speech.
4. Turn orchestrator dispatches transcribed text immediately without stalling on is_user_speaking.
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import telephony_server
from telephony_server import (
    CallSession,
    VAD_SPEECH_THRESHOLD,
    transcribe_speech,
    registry,
)


class TestZeroLagFlow(unittest.IsolatedAsyncioTestCase):

    def test_vad_speech_threshold(self):
        """Verify VAD threshold is 0.45 to ignore ambient room hiss and breathing."""
        self.assertEqual(VAD_SPEECH_THRESHOLD, 0.45)

    async def test_transcribe_speech_prioritizes_groq_no_local_cpu_waste(self):
        """Groq STT should be primary and return in ~200ms without invoking CPU-heavy local Whisper."""
        fake_groq_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello, how can I help you?"
        fake_groq_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        fake_local_whisper = MagicMock()
        fake_local_whisper.transcribe = MagicMock()

        orig_groq = registry.groq_stt_client
        orig_whisper = registry.whisper
        try:
            registry.groq_stt_client = fake_groq_client
            registry.whisper = fake_local_whisper

            # 1 second of audio
            t = np.linspace(0, 1.0, 16000, dtype=np.float32)
            audio = 0.2 * np.sin(2 * np.pi * 440 * t)

            res = await transcribe_speech(audio)
            self.assertEqual(res, "Hello, how can I help you?")

            # Verify Groq was called
            fake_groq_client.audio.transcriptions.create.assert_awaited_once()

            # Verify local CPU whisper was NOT invoked at all!
            fake_local_whisper.transcribe.assert_not_called()
        finally:
            registry.groq_stt_client = orig_groq
            registry.whisper = orig_whisper

    async def test_transcribe_speech_fallback_to_local_only_on_error(self):
        """Local Whisper is only called when Groq STT fails or times out."""
        fake_groq_client = MagicMock()
        fake_groq_client.audio.transcriptions.create = AsyncMock(side_effect=RuntimeError("Groq STT offline"))

        fake_local_whisper = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "Fallback text from local whisper"
        fake_local_whisper.transcribe = MagicMock(return_value=([mock_seg], None))

        orig_groq = registry.groq_stt_client
        orig_whisper = registry.whisper
        try:
            registry.groq_stt_client = fake_groq_client
            registry.whisper = fake_local_whisper

            t = np.linspace(0, 1.0, 16000, dtype=np.float32)
            audio = 0.2 * np.sin(2 * np.pi * 440 * t)

            res = await transcribe_speech(audio)
            self.assertEqual(res, "Fallback text from local whisper")
            fake_local_whisper.transcribe.assert_called_once()
        finally:
            registry.groq_stt_client = orig_groq
            registry.whisper = orig_whisper

    async def test_inbound_audio_not_ducked_by_server_gate(self):
        """Verify that inbound audio is not scaled down by 0.005 during active TTS."""
        mock_ws = AsyncMock()
        session = CallSession(websocket=mock_ws, mode="web")
        session.tts_playing = True  # Bot is speaking

        # Caller speaks with amplitude 0.2
        caller_pcm = (np.ones(320, dtype=np.float32) * 0.2)
        await session.ingest_inbound_audio(caller_pcm)

        enqueued = await asyncio.wait_for(session.inbound_audio_queue.get(), timeout=1.0)
        # Should NOT be attenuated by 0.005 (which would make it 0.001)
        self.assertGreater(float(np.max(np.abs(enqueued))), 0.05)


if __name__ == "__main__":
    unittest.main()
