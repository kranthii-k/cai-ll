"""
Comprehensive Test Suite for Telemetry API, Silence Trimming, and Speech-to-Text Pipeline.

Tests:
1. In-memory TelemetryTracker:
   - STT latency and transcribed text length.
   - Fast classifier stance and latency.
   - Time-to-filler (TTF) and Time-to-First-Audio (TTFA).
   - Barge-in interruption count and rate.
   - P50, P90, and P99 latency distribution summaries.
   - STT word-level accuracy (WER complement).
   - VAD duration and playback latency tracking.
2. REST Telemetry Endpoints:
   - GET /api/telemetry
   - POST /api/telemetry/reset
   - DELETE /api/telemetry
3. Audio DSP silence trimming (trim_audio_silence):
   - Trimming Kokoro padded silence at head & tail.
   - Preserving 12ms acoustic breathing envelope.
   - Handling empty and all-zero audio signals.
   - Multi-dimensional array handling.
4. STT Pipeline (transcribe_speech):
   - Empty input handling.
   - Primary Groq Whisper Cloud execution.
   - Resilient failover from Groq Cloud to Local Faster-Whisper.
   - Both backends unavailable fallback.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import sys
import time
from typing import Any, List, Optional
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from telephony_server import (
    app,
    calculate_stt_accuracy,
    registry,
    telemetry_tracker,
    TelemetryTracker,
    transcribe_speech,
    trim_audio_silence,
)


class TestTelemetryTrackerUnit(unittest.TestCase):
    """Unit tests certifying TelemetryTracker data structures, percentiles, and calculations."""

    def setUp(self):
        self.tracker = TelemetryTracker(max_history=100)

    def test_tracker_initial_empty_state(self):
        """Validates that a fresh tracker returns valid zeroed summaries without division errors."""
        summary = self.tracker.get_summary()
        self.assertEqual(summary["status"], "healthy")
        self.assertEqual(summary["total_turns"], 0)
        self.assertEqual(summary["barge_in_stats"]["total_barge_ins"], 0)
        self.assertEqual(summary["barge_in_stats"]["barge_in_rate"], 0.0)
        self.assertEqual(summary["stt_metrics"]["total_characters"], 0)
        self.assertEqual(summary["stt_metrics"]["average_accuracy"], 1.0)

        # Check that percentiles are all 0.0 for empty state
        for metric in ["stt", "classifier", "ttf", "ttfa", "ttft", "total_turnaround", "vad_duration", "playback_latency"]:
            self.assertEqual(summary["latency_summaries"][metric]["p50"], 0.0)
            self.assertEqual(summary["latency_summaries"][metric]["p90"], 0.0)
            self.assertEqual(summary["latency_summaries"][metric]["p99"], 0.0)
            self.assertEqual(summary["latency_summaries"][metric]["count"], 0)

    def test_record_single_turn_metrics(self):
        """Validates recording of a complete turn with all latency waterfall metrics."""
        turn = self.tracker.record_turn(
            turn_id=1,
            session_id="session-abc-123",
            mode="web",
            user_text="What is the forecast?",
            transcribed_text_length=22,
            stt_latency_ms=185.5,
            vad_duration_ms=1200.0,
            classifier_stance="analytical",
            classifier_latency_ms=4.2,
            time_to_filler_ms=48.0,
            time_to_first_audio_ms=195.0,
            ttft_ms=140.0,
            total_turnaround_ms=450.0,
            playback_latency_ms=80.0,
            barge_in=False,
        )

        self.assertEqual(turn["turn_id"], 1)
        self.assertEqual(turn["classifier_stance"], "analytical")
        self.assertEqual(turn["transcribed_text_length"], 22)
        self.assertEqual(turn["stt_latency_ms"], 185.5)

        summary = self.tracker.get_summary()
        self.assertEqual(summary["total_turns"], 1)
        self.assertEqual(summary["p50_summary"]["stt_ms"], 185.5)
        self.assertEqual(summary["p50_summary"]["classifier_ms"], 4.2)
        self.assertEqual(summary["p50_summary"]["ttf_ms"], 48.0)
        self.assertEqual(summary["p50_summary"]["ttfa_ms"], 195.0)
        self.assertEqual(summary["p50_summary"]["total_turnaround_ms"], 450.0)
        self.assertEqual(summary["classifier_metrics"]["stance_distribution"].get("analytical"), 1)

    def test_p50_p90_p99_percentile_mathematical_precision(self):
        """Validates that P50, P90, and P99 percentiles match exact numpy distribution values."""
        # Record 100 values from 1.0 to 100.0
        for val in range(1, 101):
            self.tracker.record_turn(
                turn_id=val,
                user_text="test text",
                stt_latency_ms=float(val),
                classifier_latency_ms=float(val) * 0.1,
                time_to_filler_ms=float(val) * 0.5,
                time_to_first_audio_ms=float(val) * 2.0,
                total_turnaround_ms=float(val) * 5.0,
            )

        summary = self.tracker.get_summary()
        stt_pct = summary["latency_summaries"]["stt"]

        # For 1..100: p50 is 50.5, p90 is 90.1, p99 is 99.01
        self.assertAlmostEqual(stt_pct["p50"], 50.5, places=1)
        self.assertAlmostEqual(stt_pct["p90"], 90.1, places=1)
        self.assertAlmostEqual(stt_pct["p99"], 99.0, places=0)
        self.assertEqual(stt_pct["count"], 100)
        self.assertAlmostEqual(stt_pct["avg"], 50.5, places=1)
        self.assertEqual(stt_pct["min"], 1.0)
        self.assertEqual(stt_pct["max"], 100.0)

    def test_barge_in_interruption_tracking_and_rate(self):
        """Validates barge-in count, event history, and interruption rate computation."""
        # 8 normal turns
        for i in range(1, 9):
            self.tracker.record_turn(turn_id=i, user_text="Hello", barge_in=False)

        # 2 barge-ins
        self.tracker.record_barge_in(session_id="sess-1", turn_id=9, mode="web")
        self.tracker.record_turn(turn_id=9, user_text="Wait stop!", barge_in=True, interrupted_by_barge_in=True)

        self.tracker.record_barge_in(session_id="sess-1", turn_id=10, mode="twilio")
        self.tracker.record_turn(turn_id=10, user_text="Hold on please", barge_in=True, interrupted_by_barge_in=True)

        summary = self.tracker.get_summary()
        self.assertEqual(summary["total_turns"], 10)
        self.assertEqual(summary["barge_in_stats"]["total_barge_ins"], 2)
        self.assertEqual(summary["barge_in_stats"]["interruption_count"], 2)
        # Barge-in rate = 2 / 10 = 0.20 (20%)
        self.assertEqual(summary["barge_in_stats"]["barge_in_rate"], 0.2)
        self.assertEqual(len(summary["barge_in_stats"]["recent_events"]), 2)

    def test_stance_distribution_tracking(self):
        """Validates conversational stance breakdown across turns."""
        stances = ["affirmative", "analytical", "affirmative", "empathetic", "analytical", "affirmative"]
        for idx, stance in enumerate(stances):
            self.tracker.record_turn(turn_id=idx + 1, classifier_stance=stance)

        summary = self.tracker.get_summary()
        dist = summary["classifier_metrics"]["stance_distribution"]
        self.assertEqual(dist["affirmative"], 3)
        self.assertEqual(dist["analytical"], 2)
        self.assertEqual(dist["empathetic"], 1)

    def test_stt_accuracy_word_error_rate_complement(self):
        """Validates calculate_stt_accuracy on various phonetic & token discrepancies."""
        # Exact match
        self.assertEqual(calculate_stt_accuracy("hello world", "hello world"), 1.0)

        # Case and punctuation insensitivity
        self.assertEqual(calculate_stt_accuracy("Hello, World!", "hello world"), 1.0)

        # 1 word difference out of 4 (75% accuracy)
        self.assertEqual(calculate_stt_accuracy("the quick brown fox", "the quick red fox"), 0.75)

        # Completely empty transcript when reference expected
        self.assertEqual(calculate_stt_accuracy("", "good morning"), 0.0)

        # Completely different words
        self.assertEqual(calculate_stt_accuracy("apple orange banana", "cat dog elephant"), 0.0)

    def test_tracker_reset(self):
        """Validates that reset() cleanly wipes all accumulated history and percentiles."""
        self.tracker.record_turn(turn_id=1, user_text="Testing reset", stt_latency_ms=100.0)
        self.tracker.record_barge_in(session_id="s1", turn_id=1)
        self.assertEqual(self.tracker.get_summary()["total_turns"], 1)

        self.tracker.reset()
        summary = self.tracker.get_summary()
        self.assertEqual(summary["total_turns"], 0)
        self.assertEqual(summary["barge_in_stats"]["total_barge_ins"], 0)
        self.assertEqual(len(summary["recent_turns"]), 0)


class TestTelemetryAPIEndpoints(unittest.TestCase):
    """Integration tests for FastAPI /api/telemetry REST endpoints."""

    def setUp(self):
        self.client = TestClient(app)
        registry.telemetry.reset()

    def test_get_telemetry_schema_and_status(self):
        """Verifies GET /api/telemetry responds with HTTP 200 and schema compliant payload."""
        response = self.client.get("/api/telemetry")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("uptime_seconds", data)
        self.assertIn("total_turns", data)
        self.assertIn("barge_in_stats", data)
        self.assertIn("stt_metrics", data)
        self.assertIn("classifier_metrics", data)
        self.assertIn("latency_summaries", data)
        self.assertIn("p50_summary", data)
        self.assertIn("p90_summary", data)
        self.assertIn("p99_summary", data)
        self.assertIn("recent_turns", data)

    def test_post_and_delete_reset_endpoint(self):
        """Verifies POST /api/telemetry/reset and DELETE /api/telemetry reset tracker metrics."""
        # Record dummy turn
        registry.telemetry.record_turn(turn_id=1, user_text="sample", stt_latency_ms=120.0)
        self.assertEqual(self.client.get("/api/telemetry").json()["total_turns"], 1)

        # Test POST reset
        res_post = self.client.post("/api/telemetry/reset")
        self.assertEqual(res_post.status_code, 200)
        self.assertEqual(res_post.json()["status"], "success")
        self.assertEqual(self.client.get("/api/telemetry").json()["total_turns"], 0)

        # Record another turn
        registry.telemetry.record_turn(turn_id=2, user_text="sample 2", stt_latency_ms=90.0)
        self.assertEqual(self.client.get("/api/telemetry").json()["total_turns"], 1)

        # Test DELETE reset
        res_del = self.client.delete("/api/telemetry")
        self.assertEqual(res_del.status_code, 200)
        self.assertEqual(res_del.json()["status"], "success")
        self.assertEqual(self.client.get("/api/telemetry").json()["total_turns"], 0)

    def test_live_turn_waterfall_in_api(self):
        """Verifies that recorded turns appear immediately in GET /api/telemetry."""
        registry.telemetry.record_turn(
            turn_id=101,
            session_id="call-live-777",
            user_text="What are your hours?",
            transcribed_text_length=21,
            stt_latency_ms=160.0,
            classifier_stance="analytical",
            classifier_latency_ms=3.5,
            time_to_filler_ms=45.0,
            time_to_first_audio_ms=180.0,
            total_turnaround_ms=410.0,
        )

        res = self.client.get("/api/telemetry")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["total_turns"], 1)
        self.assertEqual(data["p50_summary"]["ttf_ms"], 45.0)
        self.assertEqual(data["p50_summary"]["ttfa_ms"], 180.0)
        self.assertEqual(data["stt_metrics"]["total_characters"], 21)
        self.assertEqual(data["recent_turns"][0]["turn_id"], 101)


class TestSilenceTrimming(unittest.TestCase):
    """Acoustic tests verifying dead-silence removal and 12ms acoustic breathing envelope."""

    def test_trim_padded_silence_head_and_tail(self):
        """Validates removal of ~200ms dead zeros at head and tail while preserving envelope."""
        sr = 24000
        pad_samples_12ms = int(sr * 0.012)  # 288 samples

        # 200ms silence head (4800 samples)
        head_silence = np.zeros(4800, dtype=np.float32)
        # 100ms tone (2400 samples)
        t = np.linspace(0, 0.1, 2400, endpoint=False, dtype=np.float32)
        speech = 0.5 * np.sin(2 * np.pi * 440 * t)
        # 200ms silence tail (4800 samples)
        tail_silence = np.zeros(4800, dtype=np.float32)

        padded_audio = np.concatenate([head_silence, speech, tail_silence])
        trimmed = trim_audio_silence(padded_audio, threshold=0.005, pad_ms=12.0, sr=sr)

        # Expected length = speech length (2400) + 2 * pad_samples_12ms (576) = 2976
        expected_len = len(speech) + (2 * pad_samples_12ms)
        self.assertAlmostEqual(len(trimmed), expected_len, delta=5)
        self.assertLess(len(trimmed), len(padded_audio))

    def test_trim_preserves_audio_power_and_content(self):
        """Validates that non-silent speech portion is not clipped or attenuated."""
        sr = 24000
        t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False, dtype=np.float32)
        speech = 0.6 * np.sin(2 * np.pi * 500 * t)

        silence = np.zeros(2000, dtype=np.float32)
        audio = np.concatenate([silence, speech, silence])

        trimmed = trim_audio_silence(audio, threshold=0.005, pad_ms=12.0, sr=sr)

        rms_original_speech = float(np.sqrt(np.mean(speech ** 2)))
        rms_trimmed = float(np.sqrt(np.mean(trimmed ** 2)))

        # Power should be preserved within 1.0 dB
        diff_db = abs(20 * np.log10(rms_trimmed / rms_original_speech))
        self.assertLess(diff_db, 1.0)

    def test_trim_completely_silent_audio(self):
        """Validates that all-zero audio does not throw and returns cleanly."""
        all_zeros = np.zeros(16000, dtype=np.float32)
        trimmed = trim_audio_silence(all_zeros, threshold=0.005, pad_ms=12.0, sr=24000)
        self.assertIn(len(trimmed), (0, len(all_zeros)))

    def test_trim_empty_array(self):
        """Validates that an empty array returns an empty array immediately."""
        empty = np.array([], dtype=np.float32)
        trimmed = trim_audio_silence(empty, threshold=0.005, pad_ms=12.0, sr=24000)
        self.assertEqual(len(trimmed), 0)

    def test_trim_2d_squeezed_array(self):
        """Validates that multi-dimensional inputs e.g. shape (1, N) are safely processed."""
        audio_2d = np.zeros((1, 5000), dtype=np.float32)
        audio_2d[0, 1000:3000] = 0.4
        trimmed = trim_audio_silence(audio_2d, threshold=0.005, pad_ms=12.0, sr=24000)
        self.assertEqual(trimmed.ndim, 1)
        self.assertLess(len(trimmed), 5000)


class TestSTTPipeline(unittest.IsolatedAsyncioTestCase):
    """Pipeline tests verifying transcription, cloud-to-local failover, and telemetry logging."""

    async def test_transcribe_empty_audio_returns_empty_string(self):
        """Validates that 0-length audio immediately returns an empty string without model invocation."""
        result = await transcribe_speech(np.array([], dtype=np.float32))
        self.assertEqual(result, "")

    async def test_transcribe_with_groq_whisper_turbo_success(self):
        """Validates Groq Cloud Whisper Large V3 Turbo transcription path."""
        # Create mock Groq response
        mock_response = MagicMock()
        mock_response.text = "Can I speak with a representative?"

        mock_groq_client = MagicMock()
        mock_groq_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        original_groq = registry.groq_stt_client
        registry.groq_stt_client = mock_groq_client

        try:
            # 16kHz speech sample (0.1s 440Hz sine wave)
            t = np.linspace(0, 0.1, 1600, endpoint=False, dtype=np.float32)
            audio = 0.3 * np.sin(2 * np.pi * 440 * t)
            transcript = await transcribe_speech(audio)
            self.assertEqual(transcript, "Can I speak with a representative?")
            mock_groq_client.audio.transcriptions.create.assert_awaited_once()
        finally:
            registry.groq_stt_client = original_groq

    async def test_transcribe_groq_failover_to_local_whisper(self):
        """Validates seamless fallback to Local Faster-Whisper when Groq Cloud encounters an error."""
        # Mock Groq client to raise Exception
        mock_groq_client = MagicMock()
        mock_groq_client.audio.transcriptions.create = AsyncMock(
            side_effect=Exception("Groq API 503 Service Unavailable")
        )

        # Mock local Whisper model
        @dataclass
        class MockSegment:
            text: str

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = (
            [MockSegment("Local whisper fallback transcribed text")],
            None
        )

        original_groq = registry.groq_stt_client
        original_whisper = registry.whisper

        registry.groq_stt_client = mock_groq_client
        registry.whisper = mock_whisper

        try:
            t = np.linspace(0, 0.1, 1600, endpoint=False, dtype=np.float32)
            audio = 0.2 * np.sin(2 * np.pi * 440 * t)
            transcript = await transcribe_speech(audio)
            self.assertEqual(transcript, "Local whisper fallback transcribed text")
            mock_whisper.transcribe.assert_called_once()
        finally:
            registry.groq_stt_client = original_groq
            registry.whisper = original_whisper

    async def test_transcribe_speech_both_backends_none(self):
        """Validates graceful empty string return if no STT models are available."""
        original_groq = registry.groq_stt_client
        original_whisper = registry.whisper

        registry.groq_stt_client = None
        registry.whisper = None

        try:
            audio = 0.2 * np.ones(1600, dtype=np.float32)
            result = await transcribe_speech(audio)
            self.assertEqual(result, "")
        finally:
            registry.groq_stt_client = original_groq
            registry.whisper = original_whisper


def run_all_tests():
    """Runs test suite and prints structured output."""
    print("=" * 70)
    print("🚀 RUNNING TELEMETRY API & STT PIPELINE AUTOMATED TEST SUITE")
    print("=" * 70)

    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    suite.addTest(loader.loadTestsFromTestCase(TestTelemetryTrackerUnit))
    suite.addTest(loader.loadTestsFromTestCase(TestTelemetryAPIEndpoints))
    suite.addTest(loader.loadTestsFromTestCase(TestSilenceTrimming))
    suite.addTest(loader.loadTestsFromTestCase(TestSTTPipeline))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"✅ ALL {result.testsRun} TELEMETRY & STT TESTS PASSED WITH 100% RELIABILITY!")
        print("=" * 70)
        return 0
    else:
        print(f"❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
