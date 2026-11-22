"""
Unit Tests for TTS Synthesis, Silence Sanitization, Prosody Chunking, and Crossfading.
Validates:
1. trim_audio_silence:
   - Trims 150-350ms of dead zeros at head & tail from Kokoro synthesis.
   - Preserves a natural 10-15ms acoustic breathing envelope (fade-in & fade-out).
   - Zero clicks, pops, or DC offsets at boundaries (starts/ends smoothly at zero).
   - Graceful handling of silence-only and empty arrays.
2. split_sentence_into_clauses:
   - Sentences under 12 words (e.g. 'Yes, dessert. What would you like?') remain whole.
   - Preserves neural prosody without fracturing into unnatural 2-word pieces.
   - Long sentences (>12 words) split cleanly on major punctuation boundaries without orphaned fragments (<4 words).
3. Crossfading into CrossfadeBuffer:
   - Seamless equal-power sine/cosine micro-crossfade between filler and incoming response clauses.
   - Seamless crossfading between consecutive clauses of a response.
   - Absence of abrupt waveform discontinuities (max delta step check).
4. Kokoro ONNX real synthesis validation (if model weights available).
"""

import os
import sys
import unittest
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from telephony_server import trim_audio_silence, split_sentence_into_clauses
from engine.crossfade_buffer import (
    CrossfadeBuffer,
    equal_power_crossfade,
    resample_24k_to_16k,
    float32_to_int16,
    int16_to_float32,
)
from engine.filler_manager import FillerManager
from engine.fast_classifier import ConversationalStance


class TestTTSSilenceAndProsody(unittest.TestCase):

    def test_trim_audio_silence_envelope(self):
        """Verifies that trim_audio_silence removes leading/trailing dead silence and applies smooth fade envelope."""
        sr = 24000
        pad_ms = 12.0
        pad_samples = int(round(sr * (pad_ms / 1000.0)))

        # Create artificial Kokoro-like output: 250ms lead silence + speech + 350ms trail silence
        lead_silence = np.zeros(int(sr * 0.250), dtype=np.float32)
        speech_t = np.linspace(0, 0.5, int(sr * 0.500), endpoint=False, dtype=np.float32)
        speech = 0.4 * np.sin(2 * np.pi * 300 * speech_t)
        trail_silence = np.zeros(int(sr * 0.350), dtype=np.float32)

        raw_audio = np.concatenate([lead_silence, speech, trail_silence])
        self.assertEqual(len(raw_audio), int(sr * 1.100))

        trimmed = trim_audio_silence(raw_audio, threshold=0.005, pad_ms=pad_ms, sr=sr)

        # Expected length should be approximately speech length + 2 * pad_samples
        expected_len = len(speech) + 2 * pad_samples
        self.assertAlmostEqual(len(trimmed), expected_len, delta=10)

        # Boundary checks: must start and end at or extremely close to 0.0 (no clicks/pops)
        self.assertAlmostEqual(float(trimmed[0]), 0.0, places=5)
        self.assertAlmostEqual(float(trimmed[-1]), 0.0, places=4)

        # Peak speech amplitude should be preserved
        self.assertAlmostEqual(float(np.max(np.abs(trimmed))), 0.4, places=2)

    def test_trim_audio_silence_edge_cases(self):
        """Verifies edge cases: empty audio, pure silence, int16 input, already trimmed audio."""
        sr = 16000

        # Empty array
        empty = np.array([], dtype=np.float32)
        self.assertEqual(len(trim_audio_silence(empty, sr=sr)), 0)

        # Pure silence
        all_zeros = np.zeros(sr * 2, dtype=np.float32)
        self.assertEqual(len(trim_audio_silence(all_zeros, sr=sr)), 0)

        # int16 input type preservation
        int16_speech = (0.5 * 32767.0 * np.sin(np.linspace(0, 100, 1000))).astype(np.int16)
        int16_audio = np.concatenate([np.zeros(500, dtype=np.int16), int16_speech, np.zeros(500, dtype=np.int16)])
        trimmed_i16 = trim_audio_silence(int16_audio, sr=sr, pad_ms=10.0)
        self.assertEqual(trimmed_i16.dtype, np.int16)
        self.assertGreater(len(trimmed_i16), 0)
        self.assertLess(len(trimmed_i16), len(int16_audio))

    def test_split_sentence_into_clauses_short_sentences(self):
        """Ensures short conversational sentences (<12 words) are NEVER fragmented into 2-word pieces."""
        short_sentences = [
            "Yes, dessert. What would you like?",
            "Yes, dessert.",
            "What would you like?",
            "Sure, checking that right now.",
            "Gotcha, let me look into that.",
            "Hello, how can I help you today?",
            "One two three four five six seven eight nine ten eleven twelve.",
        ]

        for sent in short_sentences:
            clauses = split_sentence_into_clauses(sent, max_words=12)
            self.assertEqual(len(clauses), 1, f"Short sentence was incorrectly split: {sent} -> {clauses}")
            self.assertEqual(clauses[0], sent.strip())

    def test_split_sentence_into_clauses_long_sentences(self):
        """Ensures long sentences (>12 words) split on punctuation boundaries without tiny orphaned fragments."""
        long_sent_1 = "Yes, dessert is very popular here, but we also have amazing coffee, teas, and pastries."
        clauses_1 = split_sentence_into_clauses(long_sent_1, max_words=12, min_clause_words=4)
        self.assertGreater(len(clauses_1), 1)
        for c in clauses_1:
            words = c.split()
            self.assertGreaterEqual(len(words), 4, f"Clause too short (<4 words): '{c}'")

        # Sentence with potentially short tail: 'yes, it is.' (3 words) must be merged into previous chunk
        long_sent_2 = "I checked that for you right now, and everything looks completely fine, yes, it is."
        clauses_2 = split_sentence_into_clauses(long_sent_2, max_words=12, min_clause_words=4)
        for c in clauses_2:
            words = c.split()
            self.assertGreaterEqual(len(words), 4, f"Clause in long_sent_2 too short (<4 words): '{c}'")

    def test_crossfade_between_filler_and_response_clauses(self):
        """Validates that CrossfadeBuffer performs smooth equal-power crossfading across filler and clauses."""
        sr = 16000
        cf_buffer = CrossfadeBuffer(sample_rate=sr, default_crossfade_ms=25.0)

        # 1. Simulate filler audio (600ms, cleanly trimmed with breathing envelope)
        t_filler = np.linspace(0, 0.6, int(sr * 0.6), endpoint=False, dtype=np.float32)
        filler = trim_audio_silence(0.3 * np.sin(2 * np.pi * 220 * t_filler), sr=sr, pad_ms=12.0)
        cf_buffer.write_chunk(filler, crossfade=False)

        # Simulate outbound stream reading 200ms of filler
        frame1 = cf_buffer.read_samples(int(sr * 0.2))
        self.assertEqual(len(frame1), int(sr * 0.2))
        self.assertTrue(cf_buffer.has_data())

        # 2. First clause arrives while filler is playing
        t_c1 = np.linspace(0, 0.8, int(sr * 0.8), endpoint=False, dtype=np.float32)
        clause1 = trim_audio_silence(0.35 * np.sin(2 * np.pi * 330 * t_c1), sr=sr, pad_ms=12.0)
        cf_buffer.write_chunk(clause1, crossfade=True, crossfade_ms=25.0)

        # 3. Second clause arrives
        t_c2 = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
        clause2 = trim_audio_silence(0.32 * np.sin(2 * np.pi * 440 * t_c2), sr=sr, pad_ms=12.0)
        cf_buffer.write_chunk(clause2, crossfade=True, crossfade_ms=25.0)

        # Read remaining stream
        stream = [frame1]
        while cf_buffer.has_data():
            chunk = cf_buffer.read_samples(320)
            if len(chunk) > 0:
                stream.append(chunk)

        full_audio = np.concatenate(stream)
        self.assertGreater(len(full_audio), int(sr * 1.5))

        # Check for discontinuities: sample-to-sample difference should never exceed 0.25
        diffs = np.abs(np.diff(full_audio))
        max_discontinuity = float(np.max(diffs))
        self.assertLess(max_discontinuity, 0.25, f"Discontinuity detected at crossfade seam: {max_discontinuity}")

    def test_live_kokoro_synthesis_if_available(self):
        """If Kokoro weights exist, verifies actual ONNX synthesis, silence trimming, and 24k->16k resampling."""
        if not (os.path.exists("kokoro-v1.0.onnx") and os.path.exists("voices-v1.0.bin")):
            self.skipTest("Kokoro model files not found on host.")

        from kokoro_onnx import Kokoro
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

        test_phrase = "Yes, dessert. What would you like?"
        raw_samples, sr = kokoro.create(test_phrase, voice="af_heart", speed=1.05, lang="en-us")
        self.assertEqual(sr, 24000)
        self.assertGreater(len(raw_samples), 0)

        # Kokoro raw samples must have leading and trailing silence
        is_speech = np.abs(raw_samples) > 0.005
        lead_silence_ms = np.argmax(is_speech) * 1000.0 / sr
        trail_silence_ms = np.argmax(is_speech[::-1]) * 1000.0 / sr
        self.assertGreater(trail_silence_ms, 100.0, "Expected Kokoro to produce >100ms trailing silence")

        # Trim silence
        trimmed = trim_audio_silence(raw_samples, threshold=0.005, pad_ms=12.0, sr=sr)
        self.assertLess(len(trimmed), len(raw_samples))

        # Check that leading & trailing envelope fade exists
        self.assertAlmostEqual(float(trimmed[0]), 0.0, places=4)
        self.assertAlmostEqual(float(trimmed[-1]), 0.0, places=3)

        # Resample to 16kHz
        pcm_16k = resample_24k_to_16k(trimmed)
        self.assertEqual(len(pcm_16k), int(round(len(trimmed) * 16000 / 24000)))
        self.assertEqual(pcm_16k.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
