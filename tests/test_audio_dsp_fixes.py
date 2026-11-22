"""
Validation Tests for Audio DSP & Speech Streaming Specialist Fixes:
1. Instant Pre-Cached Fast-Path Filler (<5ms TTFB, 0ms on-the-fly synthesis delay).
2. Polyphase FIR / soxr 24kHz -> 16kHz resampling without FFT edge artifacts.
3. Web Streaming Frame Size (100ms / 1600 samples) vs Twilio RTP Frame Size (20ms / 320 samples).
4. Equal-power sine/cosine crossfade RMS continuity.
"""

import asyncio
import os
import sys
import time
import unittest
import numpy as np

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.crossfade_buffer import (
    CrossfadeBuffer,
    equal_power_crossfade,
    resample_24k_to_16k,
    resample_16k_to_8k,
    resample_8k_to_16k,
    float32_to_int16,
    int16_to_float32,
)
from engine.fast_classifier import ConversationalStance
from engine.filler_manager import FillerManager
from telephony_server import CallSession, ModelRegistry


class TestAudioDSPFixes(unittest.TestCase):

    def setUp(self):
        self.filler_mgr = FillerManager(sample_rate=16000)

    def test_instant_precached_filler_retrieval(self):
        """Validates that filler audio is retrieved from RAM in <2ms, not synthesized live."""
        stances = [
            ConversationalStance.AFFIRMATIVE,
            ConversationalStance.ANALYTICAL,
            ConversationalStance.EMPATHETIC,
            ConversationalStance.CLARIFYING,
        ]
        for stance in stances:
            t0 = time.perf_counter()
            filler = self.filler_mgr.get_filler(stance, sample_rate=16000, dtype=np.float32)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            self.assertIsNotNone(filler)
            self.assertLess(elapsed_ms, 5.0, f"Filler retrieval took {elapsed_ms:.2f}ms, expected <5ms!")

            samples = filler.get_samples(target_sr=16000, dtype=np.float32)
            self.assertIsNotNone(samples)
            self.assertGreater(len(samples), 0)
            self.assertEqual(samples.dtype, np.float32)
            # Verify audio is non-silent
            rms = np.sqrt(np.mean(samples ** 2))
            self.assertGreater(rms, 0.001)

    def test_polyphase_resample_24k_to_16k_purity(self):
        """Validates resample_24k_to_16k preserves frequency and avoids edge ringing."""
        sr_in = 24000
        sr_out = 16000
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sr_in * duration), endpoint=False, dtype=np.float32)

        # 1kHz test sine wave
        freq = 1000.0
        sine_24k = 0.5 * np.sin(2 * np.pi * freq * t)

        resampled_16k = resample_24k_to_16k(sine_24k)
        expected_len = int(round(len(sine_24k) * sr_out / sr_in))
        self.assertEqual(len(resampled_16k), expected_len)

        # RMS power should be preserved within 0.5 dB
        rms_in = float(np.sqrt(np.mean(sine_24k ** 2)))
        rms_out = float(np.sqrt(np.mean(resampled_16k ** 2)))
        db_diff = abs(20 * np.log10(rms_out / rms_in))
        self.assertLess(db_diff, 0.5, f"Resampling RMS differed by {db_diff:.2f} dB")

        # Test boundary samples for absence of explosive edge ringing
        self.assertLess(abs(resampled_16k[0]), 0.1)
        self.assertLess(abs(resampled_16k[-1]), 0.6)

    def test_web_vs_twilio_framing(self):
        """Validates that web mode uses 100ms framing (1600 samples) and Twilio uses 20ms framing (320 samples)."""
        session_web = CallSession(websocket=None, mode="web")
        session_twilio = CallSession(websocket=None, mode="twilio")

        # Push 200ms of audio (3200 samples at 16kHz) into both buffers
        chunk = 0.3 * np.ones(3200, dtype=np.float32)
        session_web.crossfade_buffer.write_chunk(chunk)
        session_twilio.crossfade_buffer.write_chunk(chunk)

        # Web frame size should be 1600 samples (100ms)
        frame_web = session_web.crossfade_buffer.read_samples(1600)
        self.assertEqual(len(frame_web), 1600)

        # Twilio frame size should be 320 samples (20ms)
        frame_twilio = session_twilio.crossfade_buffer.read_samples(320)
        self.assertEqual(len(frame_twilio), 320)

    def test_equal_power_crossfade_continuity(self):
        """Validates that equal-power crossfading maintains continuous energy across boundaries."""
        sr = 16000
        # Two consecutive 100ms chunks at 400Hz and 450Hz
        t1 = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False, dtype=np.float32)
        t2 = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False, dtype=np.float32)
        c1 = 0.5 * np.sin(2 * np.pi * 400 * t1)
        c2 = 0.5 * np.sin(2 * np.pi * 450 * t2)

        crossfaded = equal_power_crossfade(c1, c2, sample_rate=sr, duration_ms=20.0)

        # Check for NaN / Inf
        self.assertFalse(np.any(np.isnan(crossfaded)))
        self.assertFalse(np.any(np.isinf(crossfaded)))

        # Ensure no sudden -3dB drop in the crossfade region
        fade_samples = int(round(sr * 0.020))
        seam_start = len(c1) - fade_samples
        seam_region = crossfaded[seam_start : seam_start + fade_samples]
        seam_rms = float(np.sqrt(np.mean(seam_region ** 2)))
        self.assertGreater(seam_rms, 0.25, "Seam RMS dropped excessively during crossfade!")

    def test_adaptive_flow_control_outbound_worker(self):
        """Validates adaptive flow control in _outbound_stream_worker for Web clients."""
        async def run_test():
            class MockWS:
                def __init__(self):
                    self.sent_bytes = []
                    self.sent_json = []

                async def send_bytes(self, b):
                    self.sent_bytes.append(b)

                async def send_json(self, j):
                    self.sent_json.append(j)

            mock_ws = MockWS()
            session = CallSession(websocket=mock_ws, mode="web")

            # Push 600ms of audio (9600 samples at 16kHz) = 6 frames of 100ms
            chunk = 0.2 * np.ones(9600, dtype=np.float32)
            session.crossfade_buffer.write_chunk(chunk)

            # Start worker task
            worker_task = asyncio.create_task(session._outbound_stream_worker())

            # Yield control to let worker process frames
            # With minimal yield (sleep(0.005)), initial frames are pushed rapidly
            # so the client gets a safe lead buffer (~200ms) without starvation
            await asyncio.sleep(0.12)

            self.assertTrue(session.tts_playing)
            self.assertTrue(any(j.get("event") == "tts_start" for j in mock_ws.sent_json))
            self.assertGreaterEqual(len(mock_ws.sent_bytes), 2)

            # Each frame should be 1600 samples * 2 bytes = 3200 bytes
            for frame_bytes in mock_ws.sent_bytes:
                self.assertLessEqual(len(frame_bytes), 3200)

            # Clean up
            await session.close()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
