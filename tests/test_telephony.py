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
from telephony_server import (
    app,
    CallSession,
    mulaw_to_pcm16_16k,
    pcm16k_to_mulaw8k,
    DEFAULT_SILENCE_TIMEOUT_MS,
    SILENCE_TIMEOUT_MS,
    calculate_dynamic_silence_timeout,
    check_acoustic_clause_completion,
    is_incomplete_thought,
)


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

    def test_laptop_speakerphone_loud_feedback_never_triggers_false_barge_in(self):
        """Simulates loud laptop speakerphone echo (coupling factor 0.5): must never trigger false barge-in."""
        self.tap.clear()
        self.processor.reset_state()

        # Bot is speaking modulated audio
        total_frames = 40
        frame_samples = 320  # 20ms
        t_out = np.linspace(0, total_frames * 0.02, total_frames * frame_samples, endpoint=False, dtype=np.float32)
        f = 200 + 40 * np.sin(2 * np.pi * 4 * t_out)
        bot_audio = 0.6 * np.sin(2 * np.pi * f * t_out)

        # 60ms delay = 3 frames of 20ms
        delay_frames = 3

        # Prime the tap with the initial 60ms of bot speech
        for f_idx in range(delay_frames):
            frame = bot_audio[f_idx * frame_samples:(f_idx + 1) * frame_samples]
            self.tap.push_reference(frame, sample_rate=self.sample_rate)

        # Stream remaining frames: bot plays new frame, mic receives 60ms delayed echo at coupling 0.5
        false_barge_in = False
        for f_idx in range(delay_frames, delay_frames + 25):
            bot_frame = bot_audio[f_idx * frame_samples:(f_idx + 1) * frame_samples]
            self.tap.push_reference(bot_frame, sample_rate=self.sample_rate)

            echo_chunk = bot_audio[(f_idx - delay_frames) * frame_samples:(f_idx - delay_frames + 1) * frame_samples] * 0.5
            cleaned, is_barge, metrics = self.processor.process_frame(echo_chunk)
            if is_barge:
                false_barge_in = True
                break

        self.assertFalse(false_barge_in, "Loud laptop speakerphone echo triggered false barge-in!")

    def test_barge_in_holdoff_active_at_utterance_start(self):
        """Verifies 450ms holdoff suppresses onset transients at the start of any TTS utterance."""
        proc = InboundAudioProcessor(
            echo_tap=self.tap,
            sample_rate=self.sample_rate,
            barge_in_holdoff_ms=450.0,
            barge_in_urgent_energy=0.14
        )
        now = time.time()
        proc.notify_utterance_start(now)

        # Push 100ms of bot reference
        t_out = np.linspace(0, 0.1, int(self.sample_rate * 0.1), endpoint=False, dtype=np.float32)
        self.tap.push_reference(0.5 * np.sin(2 * np.pi * 350 * t_out))

        # Transient speaker click / power-on noise frame during holdoff window
        t_frame = np.linspace(0, 0.02, 320, endpoint=False, dtype=np.float32)
        transient_frame = 0.25 * np.sin(2 * np.pi * 1000 * t_frame)

        # Process frames within holdoff (<450ms)
        for _ in range(10):
            self.assertFalse(proc.should_interrupt(transient_frame), "Hold-off failed to suppress onset transient!")

        # After holdoff expires (>450ms), human caller says 'stop'
        proc._utterance_start_time = now - 0.500  # 500ms ago (>450ms)
        # Push active reference so bot is speaking
        t_active = np.linspace(0, 0.5, int(self.sample_rate * 0.5), endpoint=False, dtype=np.float32)
        self.tap.push_reference(0.4 * np.sin(2 * np.pi * 350 * t_active))

        caller_voice = 0.18 * np.sin(2 * np.pi * 600 * t_frame)
        barge_in = False
        for _ in range(8):  # 160ms urgent speech
            if proc.should_interrupt(caller_voice):
                barge_in = True
                break
        self.assertTrue(barge_in, "True barge-in failed to trigger after holdoff elapsed!")

    def test_urgent_barge_in_keywords_sub_200ms_cutoff(self):
        """Caller saying urgent keyword ('stop', 'wait', 'hold on') triggers sub-200ms interruption."""
        proc = InboundAudioProcessor(
            echo_tap=self.tap,
            sample_rate=self.sample_rate,
            barge_in_urgent_energy=0.14
        )
        # Bot is speaking
        t_out = np.linspace(0, 1.0, self.sample_rate, endpoint=False, dtype=np.float32)
        self.tap.push_reference(0.4 * np.sin(2 * np.pi * 300 * t_out))

        # Caller says 'STOP!' (duration 160ms = 8 frames, energy 0.18)
        frame_samples = 320
        t_frame = np.linspace(0, 0.02, frame_samples, endpoint=False, dtype=np.float32)
        urgent_speech = 0.18 * np.sin(2 * np.pi * 700 * t_frame)

        barge_in_triggered = False
        frames_to_trigger = 0
        for idx in range(8):
            if proc.should_interrupt(urgent_speech):
                barge_in_triggered = True
                frames_to_trigger = idx + 1
                break

        self.assertTrue(barge_in_triggered, "Urgent barge-in keyword failed to trigger!")
        latency_ms = frames_to_trigger * 20
        self.assertLessEqual(latency_ms, 180, f"Urgent barge-in took {latency_ms}ms, exceeded 180ms budget!")

    def test_double_talk_echo_cancelled_caller_audio_preserved(self):
        """Under double-talk, speaker echo is subtracted while caller's voice is preserved unmuted."""
        # Bot speaking 300Hz tone
        t_out = np.linspace(0, 0.5, int(self.sample_rate * 0.5), endpoint=False, dtype=np.float32)
        bot_audio = 0.5 * np.sin(2 * np.pi * 300 * t_out)
        self.tap.push_reference(bot_audio)

        # Delayed echo: 60ms delay, coupling 0.4
        delay_s = int(0.060 * self.sample_rate)
        echo_frame = bot_audio[delay_s:delay_s + 320] * 0.4

        # Caller says 'WAIT!': 800Hz tone, amplitude 0.20
        t_frame = np.linspace(0, 0.02, 320, endpoint=False, dtype=np.float32)
        caller_voice = 0.20 * np.sin(2 * np.pi * 800 * t_frame)

        double_talk_frame = echo_frame + caller_voice

        cleaned, should_interrupt, metrics = self.processor.process_frame(double_talk_frame)

        # In double-talk, caller voice must NOT be suppressed by 95%!
        caller_orig_rms = float(np.sqrt(np.mean(caller_voice ** 2)))
        cleaned_rms = float(np.sqrt(np.mean(cleaned ** 2)))

        self.assertGreater(cleaned_rms, 0.5 * caller_orig_rms, "Caller voice was aggressively muted in double-talk!")
        self.assertFalse(metrics.is_echo, "Double-talk was erroneously classified as pure echo!")


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

    def test_telemetry_endpoint(self):
        """Verifies GET /api/telemetry returns full latency waterfall and percentile metrics."""
        response = self.client.get("/api/telemetry")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("barge_in_stats", data)
        self.assertIn("stt_metrics", data)
        self.assertIn("latency_summaries", data)
        self.assertIn("p50_summary", data)
        self.assertIn("p90_summary", data)
        self.assertIn("p99_summary", data)
        self.assertIn("recent_turns", data)

    def test_telemetry_reset_endpoint(self):
        """Verifies POST /api/telemetry/reset resets all counters."""
        reset_res = self.client.post("/api/telemetry/reset")
        self.assertEqual(reset_res.status_code, 200)
        self.assertEqual(reset_res.json().get("status"), "success")

        get_res = self.client.get("/api/telemetry")
        self.assertEqual(get_res.status_code, 200)
        data = get_res.json()
        self.assertEqual(data.get("total_turns"), 0)
        self.assertEqual(data.get("barge_in_stats", {}).get("total_barge_ins"), 0)


class TestDynamicTurnTakingAndVAD(unittest.TestCase):
    """Tests for Dynamic VAD Endpointing, Turn-Taking, and Speech Chopping Prevention."""

    def test_default_silence_timeout_in_range(self):
        """Verifies default silence timeout is in the 450ms–500ms range."""
        self.assertGreaterEqual(DEFAULT_SILENCE_TIMEOUT_MS, 450.0)
        self.assertLessEqual(DEFAULT_SILENCE_TIMEOUT_MS, 500.0)
        self.assertEqual(SILENCE_TIMEOUT_MS, DEFAULT_SILENCE_TIMEOUT_MS)

    def test_short_speech_requires_at_least_500ms_silence(self):
        """If accumulated speech < 1.0s, require >= 500ms of silence before concluding turn."""
        # 0.4s utterance
        timeout_short = calculate_dynamic_silence_timeout(speech_duration_s=0.4, is_clause_complete=False)
        self.assertGreaterEqual(timeout_short, 500.0)

        # 0.9s utterance
        timeout_09 = calculate_dynamic_silence_timeout(speech_duration_s=0.9, is_clause_complete=True)
        self.assertGreaterEqual(timeout_09, 500.0)

    def test_long_speech_clause_complete_allows_400ms(self):
        """If speech is longer (>1.5s) and user completed a clause, allow 400ms."""
        timeout_long_clause = calculate_dynamic_silence_timeout(speech_duration_s=1.8, is_clause_complete=True)
        self.assertEqual(timeout_long_clause, 400.0)

        timeout_long_2s = calculate_dynamic_silence_timeout(speech_duration_s=2.5, is_clause_complete=True)
        self.assertEqual(timeout_long_2s, 400.0)

    def test_long_speech_incomplete_clause_uses_default_timeout(self):
        """If speech is longer (>1.5s) but clause is incomplete, use default timeout (480ms)."""
        timeout_incomplete = calculate_dynamic_silence_timeout(speech_duration_s=1.8, is_clause_complete=False)
        self.assertEqual(timeout_incomplete, DEFAULT_SILENCE_TIMEOUT_MS)

    def test_acoustic_clause_completion_cadence(self):
        """Tests acoustic clause completion detection via terminal energy decay."""
        # Case 1: Falling energy decay across the last frames (natural terminal cadence)
        frames_decay = [0.25 * np.ones(512, dtype=np.float32) for _ in range(30)]  # 30 * 32ms ~= 1.0s
        # Terminal decay over last 5 frames
        frames_decay.extend([
            0.15 * np.ones(512, dtype=np.float32),
            0.10 * np.ones(512, dtype=np.float32),
            0.06 * np.ones(512, dtype=np.float32),
            0.04 * np.ones(512, dtype=np.float32),
            0.02 * np.ones(512, dtype=np.float32),
        ])
        is_complete = check_acoustic_clause_completion(
            audio_buffer=frames_decay,
            speech_frames_count=35,
            speech_duration_s=1.8
        )
        self.assertTrue(is_complete, "Natural energy decay should indicate completed clause")

        # Case 2: Abrupt cutoff (high energy right up to silence boundary)
        frames_abrupt = [0.25 * np.ones(512, dtype=np.float32) for _ in range(35)]
        is_abrupt_complete = check_acoustic_clause_completion(
            audio_buffer=frames_abrupt,
            speech_frames_count=35,
            speech_duration_s=1.8
        )
        self.assertFalse(is_abrupt_complete, "Abrupt cutoff should NOT indicate completed clause")

    def test_syntactic_thought_completeness(self):
        """Verifies syntactic completeness checking for thoughts."""
        # Incomplete fragments
        self.assertTrue(is_incomplete_thought("Kind of weird,"))
        self.assertTrue(is_incomplete_thought("I mean,"))
        self.assertTrue(is_incomplete_thought("I mean"))
        self.assertTrue(is_incomplete_thought("because"))
        self.assertTrue(is_incomplete_thought("I wanted to check because"))
        self.assertTrue(is_incomplete_thought("and then"))
        self.assertTrue(is_incomplete_thought("so"))
        self.assertTrue(is_incomplete_thought("plan..."))

        # Complete thoughts
        self.assertFalse(is_incomplete_thought("Kind of weird, I mean, to John Doe's plan."))
        self.assertFalse(is_incomplete_thought("What time does the store open?"))
        self.assertFalse(is_incomplete_thought("I would like to cancel my subscription."))

    def test_chopping_prevention_on_human_pauses(self):
        """
        Simulates caller speaking:
        'Kind of weird' [250ms pause] 'I mean' [250ms pause] 'to John Doe' [250ms pause] 'plan' [600ms final pause]
        Verifies that with 250ms pauses, human speech is NOT chopped into fragments,
        but collected as a single complete thought before turn conclusion.
        """
        frame_ms = 32.0
        sequence = [
            (True, 600),   # 'Kind of weird'
            (False, 250),  # 250ms natural pause
            (True, 350),   # 'I mean'
            (False, 250),  # 250ms natural pause
            (True, 600),   # 'to John Doe'
            (False, 250),  # 250ms natural pause
            (True, 300),   # 'plan'
            (False, 600),  # 600ms turn-concluding silence
        ]

        emitted_speech_chunks = []
        audio_buffer = []
        pre_speech_buffer = []
        is_speaking = False
        silence_ms = 0.0
        speech_frames = 0

        for is_speech, duration_ms in sequence:
            num_frames = int(duration_ms / frame_ms)
            for _ in range(num_frames):
                frame = 0.2 * np.ones(512, dtype=np.float32) if is_speech else np.zeros(512, dtype=np.float32)
                prob = 0.85 if is_speech else 0.0

                if prob > 0.25:
                    if not is_speaking:
                        is_speaking = True
                        audio_buffer.extend(pre_speech_buffer)
                        pre_speech_buffer = []
                        speech_frames = 0
                    audio_buffer.append(frame)
                    speech_frames += 1
                    silence_ms = 0.0
                else:
                    if not is_speaking:
                        pre_speech_buffer.append(frame)
                        if len(pre_speech_buffer) > 25:
                            pre_speech_buffer.pop(0)
                    else:
                        audio_buffer.append(frame)
                        silence_ms += frame_ms
                        speech_s = speech_frames * (512 / 16000)
                        is_clause_comp = speech_s > 1.5
                        timeout = calculate_dynamic_silence_timeout(speech_s, is_clause_comp)

                        if silence_ms >= timeout:
                            is_speaking = False
                            silence_ms = 0.0
                            if len(audio_buffer) > 0:
                                emitted_speech_chunks.append(np.concatenate(audio_buffer))
                            audio_buffer = []
                            speech_frames = 0

        # Must emit EXACTLY 1 chunk containing the full utterance, NOT 4 chopped pieces!
        self.assertEqual(
            len(emitted_speech_chunks), 1,
            f"Speech was chopped into {len(emitted_speech_chunks)} fragments instead of 1 complete thought!"
        )


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
    suite.addTest(loader.loadTestsFromTestCase(TestDynamicTurnTakingAndVAD))

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
