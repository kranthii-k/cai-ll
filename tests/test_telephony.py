"""
Unit and Integration Tests for Telephony Server, EchoReferenceTap, and Full-Duplex Audio Bridge.
Run with:
    .\\venv\\Scripts\\python.exe tests/test_telephony.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
import unittest
from typing import List

import numpy as np
from fastapi.testclient import TestClient

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.crossfade_buffer import CrossfadeBuffer, apply_crossfade, apply_fade_in, apply_fade_out
from engine.echo_canceller import EchoReferenceTap, InboundAudioProcessor
from engine.fast_classifier import ConversationalStance, FastClassifier
from engine.filler_manager import FillerManager
from telephony_server import app, CallSession, mulaw_to_pcm16_16k, pcm16k_to_mulaw8k


class TestEchoReferenceTap(unittest.TestCase):
    """Tests for circular audio reference buffer."""

    def setUp(self):
        self.sample_rate = 16000
        self.tap = EchoReferenceTap(sample_rate=self.sample_rate, buffer_duration_sec=2.0)

    def test_push_and_delayed_retrieval(self):
        # Push 100ms of 440Hz sine wave
        t = np.linspace(0, 0.1, int(self.sample_rate * 0.1), endpoint=False, dtype=np.float32)
        sine = 0.5 * np.sin(2 * np.pi * 440 * t)
        self.tap.push_reference(sine, sample_rate=self.sample_rate)

        self.assertEqual(self.tap.total_samples_written, len(sine))
        self.assertTrue(self.tap.is_speaking(window_ms=100.0))

        # Query delayed reference: 40ms delay, 320 samples (20ms)
        delayed_ref = self.tap.get_delayed_reference(num_samples=320, delay_ms=40.0)
        self.assertEqual(len(delayed_ref), 320)
        # Verify it is non-zero
        self.assertGreater(float(np.sqrt(np.mean(delayed_ref ** 2))), 0.05)

    def test_circular_wrap_around(self):
        # Write more samples than buffer size to verify circular wrapping
        total_to_write = int(self.sample_rate * 3.0)  # 3s into 2s buffer
        chunk = (0.2 * np.ones(1600, dtype=np.float32))
        for _ in range(total_to_write // 1600):
            self.tap.push_reference(chunk, sample_rate=self.sample_rate)

        self.assertEqual(self.tap.total_samples_written, total_to_write)
        # Should cleanly read without index errors
        slice_out = self.tap.get_delayed_reference(num_samples=640, delay_ms=50.0)
        self.assertEqual(len(slice_out), 640)

    def test_instant_clear(self):
        self.tap.push_reference(np.ones(1600, dtype=np.float32), sample_rate=self.sample_rate)
        self.tap.clear()
        self.assertEqual(self.tap.total_samples_written, 0)
        self.assertFalse(self.tap.is_speaking())


class TestAcousticEchoCancellationAndBargeIn(unittest.TestCase):
    """Tests for Echo Cancellation, Backchannel vs Barge-in discrimination."""

    def setUp(self):
        self.sample_rate = 16000
        self.tap = EchoReferenceTap(sample_rate=self.sample_rate, buffer_duration_sec=3.0)
        self.processor = InboundAudioProcessor(
            echo_tap=self.tap,
            sample_rate=self.sample_rate,
            min_delay_ms=40.0,
            max_delay_ms=120.0,
            barge_in_min_duration_ms=300.0,
            backchannel_max_duration_ms=350.0
        )

    def test_acoustic_echo_suppression_prevents_self_interruption(self):
        """Simulates speakerphone acoustic echo: bot audio leaks back into mic with 60ms delay."""
        # Bot is speaking 440Hz tone
        t_out = np.linspace(0, 0.5, int(self.sample_rate * 0.5), endpoint=False, dtype=np.float32)
        bot_audio = 0.6 * np.sin(2 * np.pi * 440 * t_out)
        self.tap.push_reference(bot_audio, sample_rate=self.sample_rate)

        # Inbound audio is 60ms delayed echo of bot speech attenuated by coupling factor 0.3
        delay_samples = int(60.0 * self.sample_rate / 1000.0)
        echo_chunk = bot_audio[delay_samples:delay_samples + 320] * 0.3

        cleaned, is_barge_in, metrics = self.processor.process_frame(echo_chunk)

        # Must detect echo and NOT trigger false barge-in
        self.assertTrue(metrics.is_echo or metrics.cross_correlation > 0.40)
        self.assertFalse(is_barge_in)
        self.assertFalse(self.processor.should_interrupt(echo_chunk))

    def test_backchannel_discrimination(self):
        """Simulates caller backchanneling ('mhm', 'uh-huh') < 350ms duration with low energy."""
        # Bot speaking
        t_out = np.linspace(0, 0.8, int(self.sample_rate * 0.8), endpoint=False, dtype=np.float32)
        self.tap.push_reference(0.4 * np.sin(2 * np.pi * 300 * t_out))

        # Caller makes 200ms short sound (10 frames of 20ms) with low/moderate energy
        frame_samples = 320  # 20ms
        t_bc = np.linspace(0, 0.02, frame_samples, endpoint=False, dtype=np.float32)
        backchannel_frame = 0.045 * np.sin(2 * np.pi * 750 * t_bc)  # Low energy

        interrupted = False
        # Send 10 frames = 200ms (< 350ms)
        for _ in range(10):
            if self.processor.should_interrupt(backchannel_frame):
                interrupted = True
                break

        # Followed by silence (200ms)
        silence_frame = np.zeros(frame_samples, dtype=np.float32)
        for _ in range(10):
            _, is_barge, metrics = self.processor.process_frame(silence_frame)
            if is_barge:
                interrupted = True

        # Must NOT interrupt on backchannel
        self.assertFalse(interrupted, "Backchanneling erroneously triggered barge-in!")

    def test_true_barge_in_interruption(self):
        """Simulates caller speaking continuously for >300ms ('Wait stop, that is wrong!')."""
        # Bot speaking
        t_out = np.linspace(0, 1.0, int(self.sample_rate * 1.0), endpoint=False, dtype=np.float32)
        self.tap.push_reference(0.5 * np.sin(2 * np.pi * 300 * t_out))

        # Caller speaking sustained high-energy voice frames
        frame_samples = 320  # 20ms
        t_frame = np.linspace(0, 0.02, frame_samples, endpoint=False, dtype=np.float32)
        caller_speech = 0.12 * np.sin(2 * np.pi * 500 * t_frame)  # Active speech

        barge_in_fired = False
        # Send 18 frames = 360ms (> 300ms threshold)
        for frame_idx in range(18):
            if self.processor.should_interrupt(caller_speech):
                barge_in_fired = True
                break

        self.assertTrue(barge_in_fired, "Continuous speech >300ms failed to trigger barge-in!")


class TestCrossfadeBuffer(unittest.TestCase):
    """Tests for equal-power crossfader and non-blocking streaming queue."""

    def setUp(self):
        self.cf_buffer = CrossfadeBuffer(sample_rate=16000, default_crossfade_ms=30.0)

    def test_crossfade_mathematical_continuity(self):
        # Two audio arrays
        tail = np.ones(800, dtype=np.float32) * 0.5
        head = np.ones(800, dtype=np.float32) * 0.8
        overlap = 480  # 30ms

        blended = apply_crossfade(tail, head, overlap_samples=overlap, curve="equal_power")
        expected_len = len(tail) + len(head) - overlap
        self.assertEqual(len(blended), expected_len)

        # Overlap region should have smooth equal-power transition (at t=0.5, (0.5+0.8)*cos(pi/4) ~= 0.919)
        midpoint = len(tail) - (overlap // 2)
        self.assertTrue(0.45 < blended[midpoint] < 0.95)

    def test_nonblocking_queue_and_flush(self):
        samples = np.random.uniform(-0.5, 0.5, 3200).astype(np.float32)
        self.cf_buffer.write_chunk(samples)

        self.assertTrue(self.cf_buffer.has_data())
        self.assertEqual(self.cf_buffer.remaining_samples, 3200)

        # Pop 320 samples
        chunk = self.cf_buffer.read_samples(320)
        self.assertEqual(len(chunk), 320)
        self.assertEqual(self.cf_buffer.remaining_samples, 2880)

        # Immediate flush (<1ms)
        t0 = time.perf_counter()
        self.cf_buffer.flush()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(self.cf_buffer.remaining_samples, 0)
        self.assertFalse(self.cf_buffer.has_data())
        self.assertLess(elapsed_ms, 5.0, "Flush exceeded 5ms budget!")


class TestTelephonyFastAPIEndpoints(unittest.TestCase):
    """Integration tests for FastAPI REST and WebSocket endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_twiml_endpoint(self):
        response = self.client.post("/twiml", headers={"host": "voice.example.com"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/xml", response.headers["content-type"])
        self.assertIn('<Stream url="wss://voice.example.com/stream"', response.text)
        self.assertIn("<Connect>", response.text)

    def test_serve_web_phone_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Voice AI Telephony Gateway", response.text)
        self.assertIn("Time to Filler", response.text)
        self.assertIn("Barge-In Detected", response.text)

    def test_twilio_websocket_media_stream_ingestion(self):
        """Simulates Twilio Media Stream WebSocket handshake and G.711u frames."""
        with self.client.websocket_connect("/stream") as ws:
            # 1. Send Twilio 'start' event
            start_event = {
                "event": "start",
                "sequenceNumber": "1",
                "start": {
                    "streamSid": "MZ_TEST_12345",
                    "accountSid": "AC_TEST",
                    "callSid": "CA_TEST",
                    "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}
                }
            }
            ws.send_text(json.dumps(start_event))

            # 2. Send 5 frames of 8000Hz G.711u media (20ms each = 160 bytes)
            raw_mulaw = b"\xff" * 160  # Silence in mu-law
            b64_payload = base64.b64encode(raw_mulaw).decode("utf-8")

            for seq in range(2, 7):
                media_event = {
                    "event": "media",
                    "sequenceNumber": str(seq),
                    "media": {"track": "inbound", "chunk": str(seq), "payload": b64_payload}
                }
                ws.send_text(json.dumps(media_event))

            # 3. Send Twilio 'stop' event
            stop_event = {
                "event": "stop",
                "sequenceNumber": "7",
                "stop": {"accountSid": "AC_TEST", "callSid": "CA_TEST"}
            }
            ws.send_text(json.dumps(stop_event))

    def test_browser_web_stream_audio_and_ping(self):
        """Simulates browser web phone connection, ping, and 16kHz binary PCM ingestion."""
        with self.client.websocket_connect("/web_stream") as ws:
            # 1. Ping / Pong
            ws.send_text(json.dumps({"event": "ping"}))
            response = ws.receive_json()
            self.assertEqual(response.get("event"), "pong")

            # 2. Send 16kHz Int16 binary audio frames (20ms = 320 samples = 640 bytes)
            int16_frame = (np.zeros(320, dtype=np.int16)).tobytes()
            for _ in range(5):
                ws.send_bytes(int16_frame)

            # 3. Send text prompt test
            ws.send_text(json.dumps({"event": "text_prompt", "text": "What is system status?"}))

            # Expect classification event
            event = ws.receive_json()
            self.assertEqual(event.get("event"), "classification")
            self.assertIn("stance", event)
            self.assertIn("latency_ms", event)


def run_tests():
    """Runs test suite and prints structured output."""
    print("=" * 70)
    print("🚀 RUNNING FULL-DUPLEX TELEPHONY & ECHO CANCELLER TEST SUITE")
    print("=" * 70)

    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    suite.addTest(loader.loadTestsFromTestCase(TestEchoReferenceTap))
    suite.addTest(loader.loadTestsFromTestCase(TestAcousticEchoCancellationAndBargeIn))
    suite.addTest(loader.loadTestsFromTestCase(TestCrossfadeBuffer))
    suite.addTest(loader.loadTestsFromTestCase(TestTelephonyFastAPIEndpoints))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"✅ ALL {result.testsRun} TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        print(f"❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
