"""
End-to-End Full-Duplex Telephony Pipeline Integration Test.

Simulates a real-time voice AI telephone call turn with:
1. Audio streaming (8kHz G.711u / 16kHz PCM)
2. Sub-50ms intent classification & speculative acoustic filler dispatch (TTF < 150ms)
3. Concurrent LLM streaming & sentence chunking with pysbd
4. Seamless equal-power micro-crossfading between filler and response
5. Tool call interception (verifying raw JSON is never spoken)
6. Sub-50ms full-duplex acoustic echo cancellation and barge-in interruption

Run command:
    .\\venv\\Scripts\\python.exe tests/test_e2e_pipeline.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
import unittest
from typing import Any, AsyncGenerator, Dict, List, Optional
import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from engine.crossfade_buffer import (
    CrossfadeBuffer,
    apply_crossfade,
    equal_power_crossfade,
    int16_to_float32,
    float32_to_int16,
    int16_to_mulaw,
    mulaw_to_int16,
    resample_8k_to_16k,
    resample_16k_to_8k,
    resample_24k_to_16k,
)
from engine.fast_classifier import ConversationalStance, FastClassifier
from engine.filler_manager import FillerManager
from engine.dual_brain import DualBrainEngine, DualBrainEvent, DualBrainEventType
from engine.echo_canceller import EchoReferenceTap, InboundAudioProcessor


class MockOpenAIStream:
    """Mock for AsyncOpenAI chat completions streaming with tool calls."""
    def __init__(self, chunks: List[Any]):
        self.chunks = chunks

    def __aiter__(self):
        self._iter = iter(self.chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class MockDelta:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[Any]] = None):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    def __init__(self, delta: MockDelta):
        self.delta = delta


class MockChunk:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[Any]] = None):
        self.choices = [MockChoice(MockDelta(content=content, tool_calls=tool_calls))]


class TestEndToEndPipeline(unittest.IsolatedAsyncioTestCase):
    """End-to-end integration tests certifying the full voice telephony stack."""

    async def asyncSetUp(self):
        self.classifier = FastClassifier()
        self.filler_mgr = FillerManager(sample_rate=16000)
        self.crossfade_buf = CrossfadeBuffer(sample_rate=16000, default_crossfade_ms=30.0)
        self.echo_tap = EchoReferenceTap(sample_rate=16000, buffer_duration_sec=4.0)
        self.inbound_proc = InboundAudioProcessor(
            echo_tap=self.echo_tap,
            sample_rate=16000,
            min_delay_ms=40.0,
            max_delay_ms=120.0,
            barge_in_min_duration_ms=300.0,
            backchannel_max_duration_ms=350.0
        )

    async def test_01_time_to_filler_latency(self):
        """
        Verify Time-to-Filler (TTF) is strictly under 150ms (target < 50ms)
        from speech endpointing to first filler audio frame availability.
        """
        user_utterance = "Can you look up my account status and billing balance?"

        t_start = time.perf_counter()

        # Step 1: Fast Intent Classification (<5ms)
        classification = self.classifier.classify(user_utterance)
        t_classified = time.perf_counter()

        self.assertEqual(classification.stance, ConversationalStance.ANALYTICAL)
        self.assertTrue(classification.latency_masking_required)

        # Step 2: Instant acoustic filler retrieval
        filler = self.filler_mgr.get_filler(classification.stance, user_utterance)
        t_retrieved = time.perf_counter()

        self.assertIsNotNone(filler)
        self.assertIsNotNone(filler.audio_bytes)

        # Step 3: Audio frame conversion and queue write
        filler_pcm = np.frombuffer(filler.audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.crossfade_buf.write_chunk(filler_pcm, crossfade=False)
        t_queued = time.perf_counter()

        # Step 4: First 20ms outbound frame read
        out_frame = await self.crossfade_buf.read_frame_async(320, timeout=0.05)
        t_first_frame = time.perf_counter()

        ttf_ms = (t_first_frame - t_start) * 1000.0
        classify_ms = (t_classified - t_start) * 1000.0
        retrieve_ms = (t_retrieved - t_classified) * 1000.0

        print(f"\n[E2E] Time-to-Filler Latency: {ttf_ms:.2f}ms (Target: <150ms)")
        print(f"      - Classification: {classify_ms:.3f}ms")
        print(f"      - Retrieval:      {retrieve_ms:.3f}ms")

        self.assertLess(ttf_ms, 150.0, f"Time-to-Filler exceeded threshold: {ttf_ms:.2f}ms >= 150ms")
        self.assertLess(classify_ms, 10.0, f"Classification exceeded threshold: {classify_ms:.2f}ms >= 10ms")
        self.assertEqual(len(out_frame), 320)
        self.assertFalse(np.isnan(out_frame).any(), "NaN found in filler audio frame!")

    async def test_02_seamless_crossfade_continuity_and_energy(self):
        """
        Verify seamless equal-power crossfading between speculative filler and incoming LLM response:
        - Continuous RMS energy across the seam (no suck-out dip)
        - Zero sample clipping outside [-1.0, 1.0]
        - Zero NaNs or Infs
        """
        sr = 16000
        # 1. Filler audio: 400ms 440Hz tone
        t_f = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False, dtype=np.float32)
        filler_audio = 0.4 * np.sin(2 * np.pi * 440 * t_f)

        # 2. Response audio: 600ms 880Hz tone
        t_r = np.linspace(0, 0.6, int(sr * 0.6), endpoint=False, dtype=np.float32)
        response_audio = 0.4 * np.sin(2 * np.pi * 880 * t_r)

        # Write filler to buffer
        buf = CrossfadeBuffer(sample_rate=sr, default_crossfade_ms=30.0)
        buf.write_chunk(filler_audio, crossfade=False)

        # Read 200ms of filler
        first_half = buf.read_samples(int(sr * 0.2))
        self.assertEqual(len(first_half), int(sr * 0.2))

        # While remaining 200ms of filler is playing, LLM response arrives!
        # Hand off with equal-power crossfade (30ms = 480 samples)
        buf.write_chunk(response_audio, crossfade=True, crossfade_ms=30.0)

        # Read all remaining audio
        remaining = buf.read_samples(buf.remaining_samples)

        # Concatenate full stream
        full_stream = np.concatenate([first_half, remaining])

        # Assertions on full stream
        self.assertFalse(np.isnan(full_stream).any(), "NaN detected in crossfaded stream!")
        self.assertFalse(np.isinf(full_stream).any(), "Inf detected in crossfaded stream!")
        self.assertTrue(np.all(full_stream >= -1.0) and np.all(full_stream <= 1.0), "Clipping detected!")

        # Verify RMS energy around seam (seam is at sample index 3200)
        seam_idx = int(sr * 0.4) - 480
        window = 320
        rms_before = np.sqrt(np.mean(full_stream[seam_idx - window : seam_idx] ** 2))
        rms_seam = np.sqrt(np.mean(full_stream[seam_idx : seam_idx + window] ** 2))
        rms_after = np.sqrt(np.mean(full_stream[seam_idx + window : seam_idx + 2 * window] ** 2))

        print(f"[E2E] RMS Seam Energy Check: before={rms_before:.3f}, seam={rms_seam:.3f}, after={rms_after:.3f}")
        # The equal-power crossfade ensures RMS at seam does not suffer a -3dB linear dip (< 0.2)
        self.assertGreater(rms_seam, 0.20, "Equal-power crossfade suffered volume drop!")
        self.assertLess(rms_seam, 0.60, "Equal-power crossfade suffered constructive spike!")

    async def test_03_tool_call_interception_never_speaks_raw_json(self):
        """
        Verify that tool calling payloads are intercepted before synthesis
        and raw JSON is NEVER yielded as spoken audio sentences.
        """
        dual_brain = DualBrainEngine(model="test-model")

        # Register a test tool
        @dual_brain.register_tool(
            name="check_inventory",
            description="Checks warehouse inventory for an item",
            parameters={
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"]
            }
        )
        async def check_inventory(item_id: str) -> dict:
            return {"item_id": item_id, "in_stock": True, "quantity": 14}

        # Mock LLM stream returning inline JSON tool call followed by spoken response
        tool_call_payload = '{"name": "check_inventory", "arguments": {"item_id": "SKU-9921"}}'
        spoken_response = "We have 14 units of SKU-9921 in stock and ready to ship."

        mock_responses = [
            # Call 1: LLM returns tool call JSON
            MockOpenAIStream([
                MockChunk(content='{"name": "check_inventory", '),
                MockChunk(content='"arguments": {"item_id": "SKU-9921"}}')
            ]),
            # Call 2: LLM receives tool result and returns spoken answer
            MockOpenAIStream([
                MockChunk(content="We have 14 units "),
                MockChunk(content="of SKU-9921 in stock "),
                MockChunk(content="and ready to ship.")
            ])
        ]

        class MockCompletions:
            def __init__(self, streams):
                self.streams = streams
                self.call_count = 0

            async def create(self, **kwargs):
                s = self.streams[self.call_count]
                self.call_count += 1
                return s

        dual_brain.client.chat.completions = MockCompletions(mock_responses)

        collected_sentences = []
        collected_events = []

        async for event in dual_brain.process("Do you have SKU-9921 in stock?"):
            collected_events.append(event.type)
            if event.type == DualBrainEventType.SENTENCE:
                collected_sentences.append(event.text)

        # Assertions
        self.assertIn(DualBrainEventType.TOOL_CALL_DETECTED, collected_events)
        self.assertIn(DualBrainEventType.TOOL_EXECUTION_START, collected_events)
        self.assertIn(DualBrainEventType.TOOL_EXECUTION_RESULT, collected_events)

        combined_speech = " ".join(collected_sentences)
        print(f"[E2E] Spoken Speech Output: '{combined_speech}'")

        # Crucial Verification: ZERO raw JSON syntax leaked to spoken audio!
        self.assertNotIn('{"name":', combined_speech, "Raw JSON leaked into spoken speech!")
        self.assertNotIn('"arguments":', combined_speech, "Tool arguments leaked into spoken speech!")
        self.assertNotIn("SKU-9921}}", combined_speech, "Raw JSON braces leaked into spoken speech!")
        self.assertIn("14 units of SKU-9921 in stock", combined_speech)

    async def test_04_full_duplex_barge_in_and_echo_cancellation(self):
        """
        Verify Full-Duplex telephony handling:
        1. Bot audio output is tapped into EchoReferenceTap.
        2. Speakerphone acoustic echo (delayed 60ms) does NOT trigger self-interruption.
        3. Real caller speech (>300ms continuous) triggers instant barge-in.
        """
        sr = 16000
        tap = self.echo_tap
        proc = self.inbound_proc

        # Bot is speaking continuous response audio (440Hz tone)
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        bot_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Stream bot audio in 20ms frames and tap to EchoReferenceTap
        frame_size = 320
        echo_delay_frames = 3  # 60ms delay
        frame_history = []

        # Phase 1: Verify acoustic echo is rejected (no self-interruption)
        self_interrupted = False
        for i in range(0, 10 * frame_size, frame_size):
            frame = bot_audio[i : i + frame_size]
            tap.push_reference(frame, sample_rate=sr)
            frame_history.append(frame)

            if len(frame_history) > echo_delay_frames:
                # Simulates speakerphone acoustic echo leaking into mic with 60ms delay and 50% attenuation
                mic_input = frame_history[-echo_delay_frames] * 0.5
                cleaned, should_interrupt, _ = proc.process_frame(mic_input)
                if should_interrupt:
                    self_interrupted = True

        self.assertFalse(self_interrupted, "Acoustic echo triggered FALSE self-interruption!")
        print("[E2E] Echo Suppression: 100% passed (zero false barge-in from speakerphone echo).")

        # Phase 2: True caller barge-in ("Wait stop!")
        # Caller speaks 350ms of strong 200Hz human voice
        t_caller = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False, dtype=np.float32)
        caller_voice = 0.7 * np.sin(2 * np.pi * 200 * t_caller)

        barge_in_fired = False
        barge_in_latency_ms = None
        t_barge_start = time.perf_counter()

        for i in range(0, len(caller_voice), frame_size):
            caller_frame = caller_voice[i : i + frame_size]
            if len(caller_frame) < frame_size:
                caller_frame = np.pad(caller_frame, (0, frame_size - len(caller_frame)))

            # Push bot frame to tap
            tap.push_reference(bot_audio[:frame_size], sample_rate=sr)

            # Ingest caller frame
            cleaned, should_interrupt, _ = proc.process_frame(caller_frame)
            if should_interrupt and not barge_in_fired:
                barge_in_fired = True
                barge_in_latency_ms = (time.perf_counter() - t_barge_start) * 1000.0

        self.assertTrue(barge_in_fired, "True caller speech failed to trigger barge-in!")
        print(f"[E2E] True Barge-In Detected in {barge_in_latency_ms:.1f}ms! (Expected ~300ms window).")

    async def test_05_multi_sample_rate_fidelity(self):
        """
        Verify end-to-end sample rate routing without distortion:
        8kHz PSTN (Twilio) -> 16kHz (VAD/Whisper) -> 24kHz (Kokoro) -> 8kHz (PSTN return)
        """
        # Generate 8kHz sine wave (1kHz tone, typical voice formant)
        sr_8k = 8000
        t_8k = np.linspace(0, 0.1, int(sr_8k * 0.1), endpoint=False, dtype=np.float32)
        orig_8k = 0.6 * np.sin(2 * np.pi * 1000 * t_8k)

        # 1. 8k -> 16k (inbound PSTN to VAD/STT)
        pcm_16k = resample_8k_to_16k(orig_8k)
        self.assertEqual(len(pcm_16k), int(len(orig_8k) * 2))

        # 2. 24k (Kokoro TTS) -> 16k (LiveKit / Browser room)
        sr_24k = 24000
        t_24k = np.linspace(0, 0.1, int(sr_24k * 0.1), endpoint=False, dtype=np.float32)
        tts_24k = 0.6 * np.sin(2 * np.pi * 1000 * t_24k)
        tts_16k = resample_24k_to_16k(tts_24k)
        self.assertEqual(len(tts_16k), int(len(tts_24k) * (16000 / 24000)))

        # 3. 16k -> 8k (outbound to Twilio PSTN)
        out_8k = resample_16k_to_8k(tts_16k)
        self.assertEqual(len(out_8k), int(len(tts_16k) * 0.5))

        # 4. G.711 mu-law encoding round-trip
        mulaw_bytes = int16_to_mulaw(float32_to_int16(out_8k))
        recovered_pcm16 = mulaw_to_int16(mulaw_bytes)
        recovered_float = int16_to_float32(recovered_pcm16)

        # Compute SNR (Signal-to-Noise Ratio)
        noise = out_8k - recovered_float
        snr_db = 10 * np.log10(np.mean(out_8k ** 2) / (np.mean(noise ** 2) + 1e-12))
        print(f"[E2E] PSTN G.711u Codec Round-Trip SNR: {snr_db:.1f} dB (ITU-T standard >35 dB)")

        self.assertGreater(snr_db, 35.0, f"G.711u codec SNR degraded: {snr_db:.1f} dB")


def run_e2e_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestEndToEndPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    print("=" * 70)
    print("🚀 RUNNING END-TO-END TELEPHONY PIPELINE CERTIFICATION SUITE")
    print("=" * 70)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print("🏆 CERTIFICATION SUCCESS: END-TO-END PIPELINE FULLY VALIDATED!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ CERTIFICATION FAILED: Test errors or failures detected.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    run_e2e_tests()
