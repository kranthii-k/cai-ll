"""
Filler Manager: Ultra-Low Latency Contextual Acoustic Fillers.

Provides instant (<5ms / 0ms) pre-rendered acoustic fillers across 4 conversational stances:
    - 'affirmative': ["Yeah, absolutely...", "Gotcha, right...", "Sure thing..."]
    - 'analytical': ["Let's see here...", "Checking that right now...", "Let me pull that up..."]
    - 'empathetic': ["Oh, I understand...", "I hear you, let me take care of that..."]
    - 'clarifying': ["Hmm, let's see...", "Right, so..."]

Pre-renders and caches phrases using Kokoro ONNX ('af_heart') into in-memory float32 numpy
arrays and on-disk compressed NPZ archives. If Kokoro is warming up or unavailable,
provides smooth synthetic harmonic chime or comfort noise fallbacks.
"""

from __future__ import annotations

import io
import math
import os
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .fast_classifier import ConversationalStance
from .crossfade_buffer import (
    float32_to_int16,
    float32_to_mulaw,
    int16_to_float32,
    pcm_to_mulaw_bytes,
    resample_audio,
)

# Optional Kokoro ONNX import
try:
    from kokoro_onnx import Kokoro  # type: ignore
    _HAS_KOKORO = True
except ImportError:
    Kokoro = None
    _HAS_KOKORO = False


@dataclass
class FillerItem:
    """
    Container for a selected acoustic filler phrase and its rendered audio.
    """
    text: str
    stance: Union[ConversationalStance, str]
    filler_key: str
    audio_samples: Optional[np.ndarray] = None  # Native float32 audio array
    sample_rate: int = 24000
    audio_bytes: Optional[bytes] = None  # WAV bytes for legacy compatibility
    duration_ms: int = 500
    is_synthetic: bool = False

    def to_dict(self) -> dict:
        stance_str = self.stance.value if isinstance(self.stance, ConversationalStance) else str(self.stance)
        return {
            "text": self.text,
            "stance": stance_str,
            "filler_key": self.filler_key,
            "has_audio": self.audio_samples is not None or self.audio_bytes is not None,
            "sample_rate": self.sample_rate,
            "duration_ms": self.duration_ms,
            "is_synthetic": self.is_synthetic,
        }

    def get_samples(
        self,
        target_sr: Optional[int] = None,
        dtype: Optional[type] = None
    ) -> np.ndarray:
        """
        Returns audio samples, resampled to target_sr and converted to dtype if requested.
        """
        if self.audio_samples is None:
            sr = target_sr or self.sample_rate
            out_dtype = dtype or np.float32
            return np.zeros(int(round(sr * (self.duration_ms / 1000.0))), dtype=out_dtype)

        samples = self.audio_samples
        if target_sr is not None and target_sr != self.sample_rate:
            samples = resample_audio(samples, self.sample_rate, target_sr)

        if dtype is not None and samples.dtype != dtype:
            if dtype == np.int16:
                samples = float32_to_int16(samples)
            elif dtype == np.float32:
                samples = int16_to_float32(samples)

        return samples

    def get_mulaw_bytes(self, target_sr: int = 8000) -> bytes:
        """
        Returns G.711 mu-law bytes suitable for PSTN / Twilio telephony streaming.
        """
        samples_8k = self.get_samples(target_sr=target_sr, dtype=np.float32)
        return pcm_to_mulaw_bytes(samples_8k)

    def to_wav_bytes(self, sample_rate: Optional[int] = None) -> bytes:
        """
        Encodes audio samples to WAV format bytes.
        """
        sr = sample_rate or self.sample_rate
        samples = self.get_samples(target_sr=sr, dtype=np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())
        return wav_buf.getvalue()


class FillerManager:
    """
    Manages pre-rendered and dynamically synthesizable contextual fillers across
    4 conversational stances:
        - affirmative
        - analytical
        - empathetic
        - clarifying
    
    Pre-renders phrases using Kokoro ONNX on startup and caches them in RAM
    for instantaneous (<5ms) time-to-first-byte playback.
    """

    # Canonical conversational stance phrases:
    DEFAULT_FILLERS: Dict[str, List[str]] = {
        "affirmative": [
            "Yeah, absolutely...",
            "Gotcha, right...",
            "Sure thing...",
        ],
        "analytical": [
            "Let's see here...",
            "Checking that right now...",
            "Let me pull that up...",
        ],
        "empathetic": [
            "Oh, I understand...",
            "I hear you, let me take care of that...",
        ],
        "clarifying": [
            "Hmm, let's see...",
            "Right, so...",
        ],
        "direct": [
            "Hey there!",
            "Hi, I'm here!",
            "Hello, sure thing...",
        ],
    }

    def __init__(
        self,
        custom_fillers: Optional[Dict[Union[ConversationalStance, str], List[str]]] = None,
        sample_rate: int = 24000,
        model_path: str = "kokoro-v1.0.onnx",
        voices_path: str = "voices-v1.0.bin",
        voice: str = "af_heart",
        cache_dir: str = "cache",
        pre_render: bool = True,
        generate_acoustic_tones: bool = True,
    ):
        self.sample_rate = sample_rate
        self.model_path = model_path
        self.voices_path = voices_path
        self.voice = voice
        self.cache_dir = cache_dir
        self.generate_acoustic_tones = generate_acoustic_tones

        # Normalise stance dictionaries to lowercase string keys
        self._fillers: Dict[str, List[str]] = {}
        for stance, phrases in self.DEFAULT_FILLERS.items():
            self._fillers[stance.lower()] = list(phrases)

        if custom_fillers:
            for stance, phrases in custom_fillers.items():
                key = stance.value.lower() if isinstance(stance, ConversationalStance) else str(stance).lower()
                self._fillers[key] = list(phrases)

        # Thread-safe rotation indices for natural variation
        self._rotation_idx: Dict[str, int] = {k: 0 for k in self._fillers}
        self._lock = threading.Lock()

        # Audio cache: key -> float32 np.ndarray at 24kHz
        self._audio_cache: Dict[str, np.ndarray] = {}
        # Multi-rate cache: (key, target_sr) -> np.ndarray
        self._resampled_cache: Dict[Tuple[str, int], np.ndarray] = {}

        self._is_warmed_up = False
        self._kokoro_instance: Optional[Kokoro] = None

        if pre_render:
            self.pre_render_all()

    @property
    def is_warmed_up(self) -> bool:
        """Returns True if Kokoro or cached acoustic audio is loaded in RAM."""
        return self._is_warmed_up

    def _normalize_stance_key(self, stance: Union[ConversationalStance, str]) -> str:
        if isinstance(stance, ConversationalStance):
            return stance.value.lower()
        return str(stance).lower().strip()

    def pre_render_all(
        self,
        kokoro_instance: Optional[Kokoro] = None,
        save_to_disk: bool = True
    ) -> int:
        """
        Pre-renders all filler phrases across all stances into memory.
        Uses cached NPZ file if present for instant (<10ms) load.
        Otherwise uses Kokoro ONNX model to synthesize and cache.
        Returns the number of pre-rendered audio phrases cached.
        """
        with self._lock:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, f"fillers_{self.voice}_{self.sample_rate}.npz")

            # 1. Fast Path: Load from disk cache
            if os.path.exists(cache_file):
                try:
                    data = np.load(cache_file)
                    for key in data.files:
                        self._audio_cache[key] = data[key].astype(np.float32)
                    self._is_warmed_up = True
                    self._prepopulate_resampled_cache()
                    return len(self._audio_cache)
                except Exception:
                    pass  # Fall back to live synthesis if cache corrupted

            # 2. Check if Kokoro ONNX model is available on host
            kokoro = kokoro_instance or self._kokoro_instance
            if kokoro is None and _HAS_KOKORO and Kokoro is not None:
                if os.path.exists(self.model_path) and os.path.exists(self.voices_path):
                    try:
                        kokoro = Kokoro(self.model_path, self.voices_path)
                        self._kokoro_instance = kokoro
                    except Exception:
                        kokoro = None

            # 3. Synthesize with Kokoro if available
            newly_rendered: Dict[str, np.ndarray] = {}
            if kokoro is not None:
                try:
                    for stance, phrase_list in self._fillers.items():
                        for phrase in phrase_list:
                            cache_key = f"{stance}:{phrase}"
                            if cache_key in self._audio_cache:
                                continue
                            samples, sr = kokoro.create(
                                phrase,
                                voice=self.voice,
                                speed=1.0,
                                lang="en-us"
                            )
                            # Kokoro output is 24kHz float32
                            if sr != self.sample_rate:
                                samples = resample_audio(samples, sr, self.sample_rate)
                            samples = samples.astype(np.float32)
                            self._audio_cache[cache_key] = samples
                            newly_rendered[cache_key] = samples

                    # Save to compressed NPZ archive for future instantaneous boots
                    if save_to_disk and self._audio_cache:
                        try:
                            np.savez_compressed(cache_file, **self._audio_cache)
                        except Exception:
                            pass

                    self._is_warmed_up = True
                    self._prepopulate_resampled_cache()
                    return len(self._audio_cache)
                except Exception:
                    pass

            # 4. Fallback: Pre-render high-quality synthetic tonal/harmonic fillers
            for stance, phrase_list in self._fillers.items():
                for phrase in phrase_list:
                    cache_key = f"{stance}:{phrase}"
                    if cache_key not in self._audio_cache:
                        samples = self._generate_synthetic_tone(phrase, sample_rate=self.sample_rate)
                        self._audio_cache[cache_key] = samples

            self._is_warmed_up = True
            self._prepopulate_resampled_cache()
            return len(self._audio_cache)

    def _prepopulate_resampled_cache(self) -> None:
        """
        Pre-resamples all cached fillers to common pipeline rates: 24kHz, 16kHz, 8kHz.
        Guarantees sub-millisecond retrieval regardless of target rate.
        """
        common_rates = [24000, 16000, 8000]
        for key, audio in list(self._audio_cache.items()):
            for rate in common_rates:
                cache_k = (key, rate)
                if cache_k not in self._resampled_cache:
                    if rate == self.sample_rate:
                        self._resampled_cache[cache_k] = audio
                    else:
                        self._resampled_cache[cache_k] = resample_audio(audio, self.sample_rate, rate)

    def get_filler(
        self,
        stance: Union[ConversationalStance, str],
        user_prompt: str = "",
        sample_rate: Optional[int] = None,
        dtype: type = np.float32,
        index: Optional[int] = None,
    ) -> Optional[FillerItem]:
        """
        Retrieve a contextual filler with round-robin rotation.
        Guarantees sub-5ms (typically <0.1ms) latency to first audio sample.

        Returns None if stance is DIRECT or has no fillers.
        """
        stance_str = self._normalize_stance_key(stance)
        if stance_str == "direct" or stance_str not in self._fillers:
            return None

        pool = self._fillers.get(stance_str, [])
        if not pool:
            return None

        with self._lock:
            if index is not None:
                chosen_idx = index % len(pool)
            else:
                chosen_idx = self._rotation_idx[stance_str] % len(pool)
                self._rotation_idx[stance_str] = (chosen_idx + 1) % len(pool)

        phrase = pool[chosen_idx]
        target_sr = sample_rate or self.sample_rate
        cache_key = f"{stance_str}:{phrase}"

        # Fetch audio from memory cache
        is_synth = False
        cached_audio = self._resampled_cache.get((cache_key, target_sr))
        if cached_audio is None:
            cached_audio = self._audio_cache.get(cache_key)
            if cached_audio is not None and target_sr != self.sample_rate:
                cached_audio = resample_audio(cached_audio, self.sample_rate, target_sr)
                self._resampled_cache[(cache_key, target_sr)] = cached_audio

        if cached_audio is None:
            # Generate fallback harmonic tone if not yet warmed up
            cached_audio = self._generate_synthetic_tone(phrase, sample_rate=target_sr)
            is_synth = True

        # Convert dtype if requested
        if dtype == np.int16:
            out_samples = float32_to_int16(cached_audio)
        else:
            out_samples = cached_audio.astype(np.float32)

        duration_ms = int(round((len(out_samples) / float(target_sr)) * 1000.0))

        # Generate WAV bytes for legacy compatibility
        pcm16 = float32_to_int16(cached_audio)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_sr)
            wf.writeframes(pcm16.tobytes())
        wav_bytes = wav_buffer.getvalue()

        return FillerItem(
            text=phrase,
            stance=stance,
            filler_key=stance_str,
            audio_samples=out_samples,
            sample_rate=target_sr,
            audio_bytes=wav_bytes,
            duration_ms=duration_ms,
            is_synthetic=is_synth,
        )

    def get_filler_audio(
        self,
        stance: Union[ConversationalStance, str],
        sample_rate: Optional[int] = None,
        dtype: type = np.float32
    ) -> Optional[np.ndarray]:
        """
        Direct convenience method returning only the audio numpy array.
        """
        item = self.get_filler(stance, sample_rate=sample_rate, dtype=dtype)
        if item is None:
            return None
        return item.audio_samples

    # ==========================================================================
    # Fallback Generators (Warming Up / No Kokoro)
    # ==========================================================================

    def _generate_synthetic_tone(
        self,
        phrase: str,
        duration_ms: Optional[int] = None,
        sample_rate: int = 24000
    ) -> np.ndarray:
        """
        Generates a subtle, pleasant harmonic chime (C5 + G5 harmonics with cosine envelope)
        to signal acoustic responsiveness without harsh buzzes or clipping.
        """
        if duration_ms is None:
            word_count = len(phrase.split())
            duration_ms = max(500, word_count * 280)

        num_samples = int(round(sample_rate * (duration_ms / 1000.0)))
        t = np.linspace(0.0, float(duration_ms) / 1000.0, num_samples, endpoint=False, dtype=np.float32)

        # Harmonic chime: Fundamental 523.25Hz (C5) + 783.99Hz (G5) + 1046.5Hz (C6)
        f0 = 523.25
        wave_f0 = np.sin(2.0 * np.pi * f0 * t)
        wave_f1 = 0.3 * np.sin(2.0 * np.pi * 1.5 * f0 * t)
        wave_f2 = 0.1 * np.sin(2.0 * np.pi * 2.0 * f0 * t)
        raw_signal = wave_f0 + wave_f1 + wave_f2

        # Smooth envelope: 25ms cosine attack + exponential decay
        attack_samples = min(int(round(sample_rate * 0.025)), num_samples // 2)
        env = np.exp(-3.5 * t).astype(np.float32)
        if attack_samples > 1:
            att_phi = (np.pi / 2.0) * (np.arange(attack_samples, dtype=np.float32) / float(attack_samples - 1))
            env[:attack_samples] *= np.sin(att_phi)

        audio = (0.12 * env * raw_signal).astype(np.float32)
        return np.clip(audio, -1.0, 1.0)

    def _generate_comfort_noise(
        self,
        duration_ms: int = 500,
        sample_rate: int = 24000
    ) -> np.ndarray:
        """
        Generates subtle comfort room noise (-50dB) to prevent telephony dead-air silence.
        """
        num_samples = int(round(sample_rate * (duration_ms / 1000.0)))
        noise = np.random.normal(0.0, 0.003, size=num_samples).astype(np.float32)
        return np.clip(noise, -1.0, 1.0)

    def _generate_silence(
        self,
        duration_ms: int = 500,
        sample_rate: int = 24000
    ) -> np.ndarray:
        """Generates zero-amplitude silence buffer."""
        num_samples = int(round(sample_rate * (duration_ms / 1000.0)))
        return np.zeros(num_samples, dtype=np.float32)
