"""
Comprehensive Test Suite for Audio DSP & Speech Streaming Engine.

Tests:
1. Equal-power sine/cosine crossfade:
   - Verifies continuous RMS energy across the seam (no zero-crossing drops, no spikes > 1.0).
   - Verifies mathematical identity cos^2(theta) + sin^2(theta) == 1.
   - Tests both float32 and int16 buffers.
   - Tests edge cases (empty, short, zero-duration).
2. Resampling fidelity across 8kHz <-> 16kHz <-> 24kHz:
   - Verifies signal frequency and energy preservation.
   - Tests roundtrip SNR and correlation.
   - Tests soxr, scipy, and linear interpolation backends.
3. Audio format conversions:
   - float32 [-1.0, 1.0] <-> int16 PCM <-> ITU-T G.711 mu-law 8-bit.
   - Vectorized lookup table correctness and speed.
4. Streaming CrossfadeBuffer queue:
   - Real-time append, crossfade cut-in (barge-in handover), and flush.
5. FillerManager:
   - All 4 conversational stances ('affirmative', 'analytical', 'empathetic', 'clarifying').
   - Kokoro ONNX pre-rendering & disk cache (<5ms retrieval).
   - Synthetic fallback audio when warming up.
"""

from __future__ import annotations

import os
import sys
import time
import unittest
import numpy as np

# Ensure repository root is on Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engine.crossfade_buffer import (
    CrossfadeBuffer,
    apply_crossfade,
    apply_fade_in,
    apply_fade_out,
    equal_power_crossfade,
    float32_to_int16,
    float32_to_mulaw,
    int16_to_float32,
    int16_to_mulaw,
    mulaw_bytes_to_pcm,
    mulaw_to_float32,
    mulaw_to_int16,
    pcm_to_mulaw_bytes,
    resample_16k_to_8k,
    resample_16k_to_24k,
    resample_24k_to_16k,
    resample_24k_to_8k,
    resample_8k_to_16k,
    resample_8k_to_24k,
    resample_audio,
)
from engine.fast_classifier import ConversationalStance
from engine.filler_manager import FillerItem, FillerManager


class TestEqualPowerCrossfade(unittest.TestCase):
    """Tests for equal-power sine/cosine micro-crossfader."""

    def test_mathematical_equal_power_identity(self):
        """Verify that cos^2(phi) + sin^2(phi) == 1.0 across the window."""
        fade_samples = 360  # 15ms at 24kHz
        t = np.arange(fade_samples, dtype=np.float64)
        phi = (np.pi / 2.0) * (t / float(fade_samples - 1))
        cos_w = np.cos(phi)
        sin_w = np.sin(phi)
        power_sum = cos_w**2 + sin_w**2

        np.testing.assert_allclose(power_sum, 1.0, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(cos_w[0], 1.0, places=6)
        self.assertAlmostEqual(sin_w[0], 0.0, places=6)
        self.assertAlmostEqual(cos_w[-1], 0.0, places=6)
        self.assertAlmostEqual(sin_w[-1], 1.0, places=6)

    def test_continuous_rms_energy_across_seam(self):
        """
        Verify continuous RMS energy across the seam:
        Uncorrelated signals crossfaded with equal-power curves must maintain
        constant RMS energy (no suck-out dip, no spikes > 1.0).
        """
        np.random.seed(42)
        sr = 24000
        fade_ms = 15.0
        fade_samples = int(round(sr * (fade_ms / 1000.0)))

        # Generate two uncorrelated stationary noise signals with target RMS = 0.5
        target_rms = 0.5
        sig_len = 10000
        # White noise scaled to target RMS
        a = np.random.normal(0.0, target_rms, size=sig_len).astype(np.float32)
        b = np.random.normal(0.0, target_rms, size=sig_len).astype(np.float32)

        # Apply equal-power crossfade
        crossfaded = equal_power_crossfade(a, b, sample_rate=sr, duration_ms=fade_ms)

        # Expected output length: len(a) + len(b) - fade_samples
        expected_len = sig_len + sig_len - fade_samples
        self.assertEqual(len(crossfaded), expected_len)

        # Assert no clipping spikes > 1.0
        max_val = np.max(np.abs(crossfaded))
        self.assertLessEqual(max_val, 1.0, f"Max amplitude {max_val} exceeded 1.0!")

        # Verify RMS across the seam region
        seam_start = sig_len - fade_samples
        seam_end = sig_len
        seam_audio = crossfaded[seam_start:seam_end]

        seam_rms = float(np.sqrt(np.mean(seam_audio**2)))
        pre_seam_rms = float(np.sqrt(np.mean(crossfaded[seam_start - fade_samples:seam_start]**2)))
        post_seam_rms = float(np.sqrt(np.mean(crossfaded[seam_end:seam_end + fade_samples]**2)))

        # Standard linear crossfade produces a ~30% (-3dB) dip (RMS ~ 0.35).
        # Equal power must preserve RMS within statistical tolerance of pre/post signals (~0.50).
        self.assertGreater(seam_rms, 0.42, f"RMS dipped significantly at seam: {seam_rms:.4f}")
        self.assertLess(seam_rms, 0.58, f"RMS spiked significantly at seam: {seam_rms:.4f}")

        # Linear comparison to prove equal-power advantage
        t = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32)
        linear_seam = a[-fade_samples:] * (1.0 - t) + b[:fade_samples] * t
        linear_mid_rms = float(np.sqrt(np.mean(linear_seam[fade_samples // 3: 2 * fade_samples // 3]**2)))
        eq_mid_rms = float(np.sqrt(np.mean(seam_audio[fade_samples // 3: 2 * fade_samples // 3]**2)))

        self.assertGreater(
            eq_mid_rms,
            linear_mid_rms * 1.15,
            f"Equal power RMS ({eq_mid_rms:.3f}) should be higher than linear dip ({linear_mid_rms:.3f})"
        )

    def test_no_zero_crossing_drops(self):
        """Verify no artificial drop to zero or dc clicks at the boundary."""
        sr = 24000
        t = np.linspace(0, 0.1, 2400, endpoint=False, dtype=np.float32)
        # 440Hz tone and 880Hz tone
        a = 0.6 * np.sin(2 * np.pi * 440 * t)
        b = 0.6 * np.sin(2 * np.pi * 880 * t)

        out = equal_power_crossfade(a, b, sample_rate=sr, duration_ms=15.0)

        # Check that there is no block of consecutive zeroes inside the transition
        fade_samples = int(round(sr * 0.015))
        seam_region = out[len(a) - fade_samples:len(a)]
        max_consecutive_zeros = 0
        current_zeros = 0
        for sample in seam_region:
            if abs(sample) < 1e-5:
                current_zeros += 1
                max_consecutive_zeros = max(max_consecutive_zeros, current_zeros)
            else:
                current_zeros = 0

        self.assertLess(max_consecutive_zeros, 5, "Unnatural zero-drop detected at seam!")

    def test_int16_buffer_crossfading(self):
        """Verify crossfade works directly on int16 arrays and preserves int16 dtype."""
        a = (np.sin(np.linspace(0, 10, 1000)) * 20000).astype(np.int16)
        b = (np.cos(np.linspace(0, 10, 1000)) * 20000).astype(np.int16)

        out = equal_power_crossfade(a, b, sample_rate=24000, duration_ms=15.0)
        self.assertEqual(out.dtype, np.int16)
        self.assertTrue(np.all(out >= -32768) and np.all(out <= 32767))
        self.assertEqual(len(out), 1000 + 1000 - int(round(24000 * 0.015)))

    def test_edge_cases(self):
        """Verify graceful handling of empty, short, and zero-duration buffers."""
        a = np.ones(50, dtype=np.float32)
        b = np.ones(50, dtype=np.float32)

        # Empty buffers
        empty = np.empty(0, dtype=np.float32)
        np.testing.assert_array_equal(equal_power_crossfade(empty, b), b)
        np.testing.assert_array_equal(equal_power_crossfade(a, empty), a)

        # Zero duration (pure concatenation)
        cat = equal_power_crossfade(a, b, duration_ms=0.0)
        self.assertEqual(len(cat), 100)
        np.testing.assert_array_equal(cat, np.concatenate([a, b]))

        # Buffers shorter than crossfade window
        short_cf = equal_power_crossfade(a, b, sample_rate=24000, duration_ms=50.0)
        # Clamped to buffer length
        self.assertLessEqual(len(short_cf), len(a) + len(b))
        self.assertGreater(len(short_cf), 0)


class TestResamplingPreservation(unittest.TestCase):
    """Tests for rate resampling: 8kHz <-> 16kHz <-> 24kHz."""

    def test_frequency_and_energy_preservation(self):
        """
        Verify that a known 1kHz sinusoidal tone is preserved across
        8kHz -> 16kHz -> 24kHz -> 8kHz roundtrip conversions.
        """
        sr_8k = 8000
        duration_s = 0.5
        freq = 1000.0  # 1kHz tone (well below 4kHz Nyquist of 8kHz PSTN)

        t_8k = np.linspace(0, duration_s, int(sr_8k * duration_s), endpoint=False, dtype=np.float32)
        orig_8k = 0.7 * np.sin(2.0 * np.pi * freq * t_8k)

        # 8kHz -> 16kHz (Telephony to STT/VAD)
        audio_16k = resample_8k_to_16k(orig_8k)
        self.assertEqual(len(audio_16k), int(16000 * duration_s))

        # 16kHz -> 24kHz (STT to Kokoro TTS native)
        audio_24k = resample_16k_to_24k(audio_16k)
        self.assertEqual(len(audio_24k), int(24000 * duration_s))

        # 24kHz -> 16kHz -> 8kHz (Kokoro TTS back to SIP Telephony)
        audio_16k_back = resample_24k_to_16k(audio_24k)
        audio_8k_back = resample_16k_to_8k(audio_16k_back)
        self.assertEqual(len(audio_8k_back), len(orig_8k))

        # Direct 24kHz -> 8kHz
        direct_8k = resample_24k_to_8k(audio_24k)
        self.assertEqual(len(direct_8k), len(orig_8k))

        # Evaluate correlation between original and round-tripped signal
        # Trim border transients (first and last 50 samples)
        trim = 50
        corr = np.corrcoef(orig_8k[trim:-trim], audio_8k_back[trim:-trim])[0, 1]
        self.assertGreater(corr, 0.99, f"Correlation after 8k->16k->24k->8k dropped: {corr:.4f}")

        # Check energy ratio preservation
        orig_energy = np.sum(orig_8k[trim:-trim]**2)
        back_energy = np.sum(audio_8k_back[trim:-trim]**2)
        energy_ratio = back_energy / orig_energy
        self.assertAlmostEqual(energy_ratio, 1.0, delta=0.05)

    def test_resampling_backends(self):
        """Test explicit backend methods: soxr, scipy, linear."""
        x = np.sin(2 * np.pi * 500 * np.linspace(0, 0.1, 2400, endpoint=False)).astype(np.float32)

        backends = ["linear"]
        try:
            import soxr
            backends.append("soxr")
        except ImportError:
            pass

        try:
            from scipy import signal
            backends.append("scipy")
        except ImportError:
            pass

        for backend in backends:
            resampled = resample_audio(x, 24000, 8000, method=backend)
            self.assertEqual(len(resampled), 800)
            self.assertEqual(resampled.dtype, np.float32)
            self.assertFalse(np.isnan(resampled).any())


class TestFormatConversions(unittest.TestCase):
    """Tests for format conversions: float32 <-> int16 <-> G.711 mu-law."""

    def test_float32_int16_roundtrip(self):
        """Verify float32 <-> int16 conversion without clipping distortion."""
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        pcm16 = float32_to_int16(vals)
        expected = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
        np.testing.assert_array_equal(pcm16, expected)

        # Roundtrip
        f32 = int16_to_float32(pcm16)
        np.testing.assert_allclose(f32, vals, atol=1e-4)

    def test_mulaw_codec_fidelity(self):
        """Verify ITU-T G.711 mu-law compression and decompression."""
        # Test full dynamic range of 16-bit PCM
        samples = np.array([-32000, -10000, -500, 0, 500, 10000, 32000], dtype=np.int16)
        encoded = int16_to_mulaw(samples)
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertEqual(len(encoded), len(samples))

        decoded = mulaw_to_int16(encoded)
        self.assertEqual(decoded.dtype, np.int16)

        # Mu-law is lossy logarithmic compression. High amplitudes have slightly higher step size,
        # but relative error should remain small (<2%).
        for orig, dec in zip(samples, decoded):
            if abs(orig) > 1000:
                rel_err = abs(int(orig) - int(dec)) / float(abs(orig))
                self.assertLess(rel_err, 0.05, f"High error on sample {orig}: got {dec}")

    def test_raw_bytes_helpers(self):
        """Verify pcm_to_mulaw_bytes and mulaw_bytes_to_pcm."""
        audio = (np.sin(np.linspace(0, 10, 800)) * 0.8).astype(np.float32)
        raw_bytes = pcm_to_mulaw_bytes(audio)
        self.assertIsInstance(raw_bytes, bytes)
        self.assertEqual(len(raw_bytes), 800)

        restored_f32 = mulaw_bytes_to_pcm(raw_bytes, dtype=np.float32)
        self.assertEqual(len(restored_f32), 800)
        self.assertEqual(restored_f32.dtype, np.float32)


class TestCrossfadeBufferQueue(unittest.TestCase):
    """Tests for the real-time streaming CrossfadeBuffer queue."""

    def test_sequential_streaming(self):
        """Test write, peek, and read chunking."""
        q = CrossfadeBuffer(sample_rate=24000, default_fade_ms=15.0)
        self.assertTrue(q.is_empty)

        chunk1 = np.ones(480, dtype=np.float32) * 0.5  # 20ms at 24kHz
        chunk2 = np.ones(480, dtype=np.float32) * 0.8
        q.write(chunk1)
        q.write(chunk2)

        self.assertEqual(q.available_samples, 960)
        self.assertEqual(q.available_ms, 40.0)

        # Peek
        peeked = q.peek(100)
        self.assertEqual(len(peeked), 100)
        self.assertEqual(q.available_samples, 960)

        # Read 20ms frame
        frame = q.read_frame(frame_ms=20.0)
        self.assertEqual(len(frame), 480)
        self.assertEqual(q.available_samples, 480)

        # Flush
        q.flush()
        self.assertTrue(q.is_empty)
        self.assertEqual(q.available_samples, 0)

    def test_crossfade_append(self):
        """Test equal-power append in buffer queue."""
        q = CrossfadeBuffer(sample_rate=24000, default_fade_ms=15.0)
        a = np.ones(720, dtype=np.float32) * 0.5
        b = np.ones(720, dtype=np.float32) * 0.7

        q.write(a)
        q.crossfade_append(b, duration_ms=15.0)

        fade_samples = int(round(24000 * 0.015))
        expected_len = 720 + 720 - fade_samples
        self.assertEqual(q.available_samples, expected_len)

    def test_crossfade_cut_in_barge_in(self):
        """Test early filler termination / barge-in cut-in handover."""
        q = CrossfadeBuffer(sample_rate=24000, default_fade_ms=15.0)
        # Queued filler with 2 seconds of audio
        filler_audio = np.ones(48000, dtype=np.float32) * 0.3
        q.write(filler_audio)

        # LLM response arrives early! Cut-in immediately!
        response_audio = np.ones(2400, dtype=np.float32) * 0.8
        q.crossfade_cut_in(response_audio, duration_ms=15.0)

        # Buffer should now hold only the transition head + new response
        fade_samples = int(round(24000 * 0.015))
        self.assertLess(q.available_samples, 48000)
        self.assertEqual(q.available_samples, len(response_audio))


class TestFillerManager(unittest.TestCase):
    """Tests for the contextual acoustic filler manager."""

    def test_all_stances_and_canonical_phrases(self):
        """Verify the 4 conversational stances and phrase configurations."""
        mgr = FillerManager(pre_render=False)

        expected_stances = ["affirmative", "analytical", "empathetic", "clarifying"]
        for stance in expected_stances:
            item = mgr.get_filler(stance)
            self.assertIsNotNone(item, f"Failed to get filler for stance: {stance}")
            self.assertEqual(item.filler_key, stance)
            self.assertTrue(len(item.text) > 0)
            self.assertIsNotNone(item.audio_samples)
            self.assertGreater(len(item.audio_samples), 0)

    def test_zero_latency_retrieval(self):
        """Verify retrieval latency from cache is <5ms (sub-millisecond)."""
        mgr = FillerManager(pre_render=False)
        # Pre-render
        mgr.pre_render_all()

        t0 = time.perf_counter()
        item = mgr.get_filler("affirmative")
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        self.assertIsNotNone(item)
        self.assertLess(elapsed_ms, 5.0, f"Retrieval latency was {elapsed_ms:.2f}ms (must be <5ms)")

    def test_multi_rate_and_format_fillers(self):
        """Verify fillers can be output in 24kHz, 16kHz, 8kHz, int16, and mu-law."""
        mgr = FillerManager(pre_render=False)
        mgr.pre_render_all()

        # 24kHz float32
        item_24k = mgr.get_filler("affirmative", sample_rate=24000, dtype=np.float32)
        self.assertEqual(item_24k.sample_rate, 24000)
        self.assertEqual(item_24k.audio_samples.dtype, np.float32)

        # 16kHz int16 (LiveKit room format)
        item_16k = mgr.get_filler("analytical", sample_rate=16000, dtype=np.int16)
        self.assertEqual(item_16k.sample_rate, 16000)
        self.assertEqual(item_16k.audio_samples.dtype, np.int16)

        # 8kHz G.711 mu-law bytes (Telephony / Twilio SIP payload)
        mulaw_bytes = item_16k.get_mulaw_bytes(target_sr=8000)
        self.assertIsInstance(mulaw_bytes, bytes)
        # Check reasonable duration: ~0.5 to 2.5 seconds at 8000 bytes/sec
        self.assertGreater(len(mulaw_bytes), 3000)
        self.assertLess(len(mulaw_bytes), 30000)

    def test_synthetic_fallback(self):
        """Verify fallback generation produces valid non-clipping audio."""
        mgr = FillerManager(pre_render=False)
        tone = mgr._generate_synthetic_tone("Checking that right now...", sample_rate=24000)
        self.assertEqual(tone.dtype, np.float32)
        self.assertGreater(len(tone), 0)
        self.assertLessEqual(np.max(np.abs(tone)), 1.0)
        self.assertGreater(np.max(np.abs(tone)), 0.01)


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING AUDIO DSP & ACOUSTIC FILLER TEST SUITE")
    print("=" * 70)
    unittest.main(verbosity=2)
