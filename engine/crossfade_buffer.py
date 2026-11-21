"""
Crossfade Buffer & DSP Utilities for Ultra-Low Latency Speech Streaming.

Core DSP Capabilities:
- Equal-power sine/cosine micro-crossfader:
    y(t) = filler(t) * cos(pi*t / (2*T)) + response(t) * sin(pi*t / (2*T))
  Maintains constant acoustic RMS energy across stream boundaries, eliminating clicks,
  pops, and the -3dB suck-out dip of linear crossfading.
- Smooth concatenation of audio buffers (numpy float32 and int16).
- Parameterizable crossfade duration (default 15ms).
- Resampling between 8kHz (Twilio G.711u), 16kHz (VAD/Whisper), and 24kHz (Kokoro TTS)
  using `soxr` with `scipy.signal.resample_poly` and linear fallbacks.
- Zero-copy / vectorized format conversions:
    float32 [-1.0, 1.0] <-> int16 PCM [-32768, 32767] <-> ITU-T G.711 mu-law 8-bit.
- Streaming CrossfadeBuffer queue with sub-50ms instant barge-in cut-in and flush.
"""

from __future__ import annotations

import asyncio
import math
import threading
from typing import List, Optional, Tuple, Union
import numpy as np

# Optional DSP acceleration backends
try:
    import soxr  # type: ignore
    _HAS_SOXR = True
except ImportError:
    soxr = None
    _HAS_SOXR = False

try:
    from scipy import signal as scipy_signal  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    scipy_signal = None
    _HAS_SCIPY = False

try:
    import audioop  # type: ignore
    _HAS_AUDIOOP = True
except ImportError:
    audioop = None
    _HAS_AUDIOOP = False


# ==============================================================================
# 1. G.711 MU-LAW CODEC TABLES (VECTORIZED O(1) ZERO-COPY LOOKUP)
# ==============================================================================

def _build_g711_tables() -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs high-performance 64KB O(1) vectorized lookup tables for ITU-T G.711 mu-law.
    Returns:
        l2u: uint8 array of size 65536, indexed by int16.view(np.uint16)
        u2l: int16 array of size 256, indexed by mu-law uint8 byte
    """
    all_int16 = np.arange(-32768, 32768, dtype=np.int16)

    if _HAS_AUDIOOP and audioop is not None:
        try:
            raw_ulaw = np.frombuffer(audioop.lin2ulaw(all_int16.tobytes(), 2), dtype=np.uint8)
            l2u = np.zeros(65536, dtype=np.uint8)
            l2u[all_int16.view(np.uint16)] = raw_ulaw
            u2l = np.frombuffer(audioop.ulaw2lin(bytes(range(256)), 2), dtype=np.int16)
            return l2u, u2l
        except Exception:
            pass

    # Pure algorithmic ITU-T G.711 fallback table builder
    BIAS = 0x84
    seg_end = [0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF]

    def _sample_to_ulaw(sample: int) -> int:
        if sample < 0:
            sample = BIAS - sample
            mask = 0x7F
        else:
            sample += BIAS
            mask = 0xFF
        if sample > 0x7FFF:
            sample = 0x7FFF
        seg = 8
        for i, s in enumerate(seg_end):
            if sample <= s:
                seg = i
                break
        if seg >= 8:
            return (0x7F ^ mask) & 0xFF
        else:
            uval = (seg << 4) | ((sample >> (seg + 3)) & 0x0F)
            return (uval ^ mask) & 0xFF

    def _ulaw_to_sample(u_byte: int) -> int:
        u = ~u_byte & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        sample = ((mantissa << 3) + BIAS) << exponent
        sample -= BIAS
        return -sample if sign else sample

    l2u = np.zeros(65536, dtype=np.uint8)
    for s in all_int16:
        l2u[np.uint16(s)] = _sample_to_ulaw(int(s))

    u2l = np.zeros(256, dtype=np.int16)
    for i in range(256):
        u2l[i] = _ulaw_to_sample(i)

    return l2u, u2l


_INT16_TO_MULAW, _MULAW_TO_INT16 = _build_g711_tables()


# ==============================================================================
# 2. AUDIO FORMAT CONVERSIONS
# ==============================================================================

def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    Converts normalized float32 audio [-1.0, 1.0] to signed 16-bit PCM [-32768, 32767].
    Guarantees no clipping overflow or integer wrapping.
    """
    if audio.dtype == np.int16:
        return audio.copy()
    clamped = np.clip(audio, -1.0, 1.0)
    scaled = clamped * 32768.0
    return np.clip(scaled, -32768.0, 32767.0).astype(np.int16)


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """
    Converts signed 16-bit PCM audio [-32768, 32767] to normalized float32 [-1.0, 1.0].
    """
    if audio.dtype == np.float32:
        return audio.copy()
    return (audio.astype(np.float32) / 32768.0).astype(np.float32)


def int16_to_mulaw(audio: np.ndarray) -> np.ndarray:
    """
    Encodes signed 16-bit PCM array to ITU-T G.711 mu-law 8-bit unsigned integers.
    Vectorized O(1) table lookup (<0.02ms per audio second).
    """
    if audio.dtype != np.int16:
        audio = float32_to_int16(audio)
    return _INT16_TO_MULAW[audio.view(np.uint16)]


def mulaw_to_int16(mulaw: Union[np.ndarray, bytes, bytearray]) -> np.ndarray:
    """
    Decodes ITU-T G.711 mu-law 8-bit unsigned integers / bytes to signed 16-bit PCM array.
    """
    if isinstance(mulaw, (bytes, bytearray)):
        idx = np.frombuffer(mulaw, dtype=np.uint8)
    else:
        idx = np.asarray(mulaw, dtype=np.uint8)
    return _MULAW_TO_INT16[idx]


def float32_to_mulaw(audio: np.ndarray) -> np.ndarray:
    """
    Encodes normalized float32 audio [-1.0, 1.0] directly to G.711 mu-law uint8 array.
    """
    pcm16 = float32_to_int16(audio)
    return int16_to_mulaw(pcm16)


def mulaw_to_float32(mulaw: Union[np.ndarray, bytes, bytearray]) -> np.ndarray:
    """
    Decodes G.711 mu-law 8-bit to normalized float32 [-1.0, 1.0].
    """
    pcm16 = mulaw_to_int16(mulaw)
    return int16_to_float32(pcm16)


def pcm_to_mulaw_bytes(audio: np.ndarray) -> bytes:
    """
    Encodes int16 or float32 audio array directly to G.711 mu-law raw bytes (PSTN / Twilio payload).
    """
    if audio.dtype != np.int16:
        audio = float32_to_int16(audio)
    return _INT16_TO_MULAW[audio.view(np.uint16)].tobytes()


def mulaw_bytes_to_pcm(data: bytes, dtype: type = np.float32) -> np.ndarray:
    """
    Decodes raw G.711 mu-law bytes to numpy array of either np.float32 or np.int16.
    """
    pcm16 = mulaw_to_int16(data)
    if dtype == np.int16:
        return pcm16
    return int16_to_float32(pcm16)


# ==============================================================================
# 3. HIGH-PRECISION AUDIO RESAMPLING (8kHz <-> 16kHz <-> 24kHz)
# ==============================================================================

def resample_audio(
    audio: np.ndarray,
    in_rate: int,
    out_rate: int,
    method: Optional[str] = None
) -> np.ndarray:
    """
    Resamples 1D audio array from in_rate to out_rate.
    Preserves input dtype (float32 or int16).
    Backend priority:
        1. 'soxr' (libsoxr fast pristine bandlimited resampler)
        2. 'scipy' (Polyphase FIR filtering via scipy.signal.resample_poly)
        3. 'linear' (Numpy 1D linear interpolation fallback)
    """
    if in_rate == out_rate or len(audio) == 0:
        return audio.copy()

    orig_dtype = audio.dtype
    work_audio = int16_to_float32(audio) if orig_dtype == np.int16 else audio.astype(np.float32)

    chosen_method = method
    if chosen_method is None:
        if _HAS_SOXR:
            chosen_method = "soxr"
        elif _HAS_SCIPY:
            chosen_method = "scipy"
        else:
            chosen_method = "linear"

    resampled: np.ndarray

    if chosen_method == "soxr":
        if not _HAS_SOXR:
            raise RuntimeError("soxr is not installed. Fall back to 'scipy' or 'linear'.")
        resampled = soxr.resample(work_audio, in_rate, out_rate, quality="HQ").astype(np.float32)

    elif chosen_method == "scipy":
        if not _HAS_SCIPY:
            raise RuntimeError("scipy is not installed. Fall back to 'linear'.")
        gcd = math.gcd(in_rate, out_rate)
        up = out_rate // gcd
        down = in_rate // gcd
        resampled = scipy_signal.resample_poly(work_audio, up, down).astype(np.float32)

    elif chosen_method == "linear":
        target_len = int(round(len(work_audio) * (out_rate / float(in_rate))))
        orig_indices = np.linspace(0.0, float(len(work_audio)), num=len(work_audio), endpoint=False)
        target_indices = np.linspace(0.0, float(len(work_audio)), num=target_len, endpoint=False)
        resampled = np.interp(target_indices, orig_indices, work_audio).astype(np.float32)

    else:
        raise ValueError(f"Unknown resampling method '{method}'. Choose 'soxr', 'scipy', or 'linear'.")

    if orig_dtype == np.int16:
        return float32_to_int16(resampled)
    return resampled


# Specialized shortcuts for low-latency telephony & speech pipeline:
def resample_8k_to_16k(audio: np.ndarray, method: Optional[str] = None) -> np.ndarray:
    """Upsample telephony 8kHz (PSTN) to 16kHz (VAD / Whisper STT)."""
    return resample_audio(audio, 8000, 16000, method=method)


def resample_16k_to_8k(audio: np.ndarray, method: Optional[str] = None) -> np.ndarray:
    """Downsample 16kHz (Agent / STT) to 8kHz (SIP / Twilio G.711u)."""
    return resample_audio(audio, 16000, 8000, method=method)


def resample_16k_to_24k(audio: np.ndarray, method: Optional[str] = None) -> np.ndarray:
    """Upsample 16kHz audio to 24kHz (Kokoro TTS native)."""
    return resample_audio(audio, 16000, 24000, method=method)


def resample_24k_to_16k(audio: np.ndarray, method: Optional[str] = None) -> np.ndarray:
    """Downsample 24kHz (Kokoro TTS) to 16kHz (LiveKit agent room)."""
    return resample_audio(audio, 24000, 16000, method=method)


def resample_24k_to_8k(audio: np.ndarray, method: Optional[str] = None) -> np.ndarray:
    """Downsample 24kHz (Kokoro TTS) directly to 8kHz (SIP telephony)."""
    return resample_audio(audio, 24000, 8000, method=method)


def resample_8k_to_24k(audio: np.ndarray, method: Optional[str] = None) -> np.ndarray:
    """Upsample 8kHz telephony audio to 24kHz."""
    return resample_audio(audio, 8000, 24000, method=method)


# ==============================================================================
# 4. EQUAL-POWER SINE/COSINE MICRO-CROSSFADER
# ==============================================================================

def equal_power_crossfade(
    filler: np.ndarray,
    response: np.ndarray,
    sample_rate: int = 24000,
    duration_ms: float = 15.0,
) -> np.ndarray:
    """
    Concatenates two audio buffers with an equal-power sine/cosine micro-crossfade:
        y(t) = filler(t) * cos(pi*t / (2*T)) + response(t) * sin(pi*t / (2*T))

    Because cos^2(theta) + sin^2(theta) == 1, this maintains constant acoustic RMS
    energy across the seam, completely eliminating clicks, pops, and the ~3dB volume
    dip caused by standard linear crossfades.

    Supports both float32 [-1.0, 1.0] and int16 [-32768, 32767] buffers.
    Output dtype matches input dtype.

    Args:
        filler: Outgoing audio buffer (e.g. acoustic filler ending).
        response: Incoming audio buffer (e.g. agent response starting).
        sample_rate: Audio sampling frequency in Hz (default: 24000).
        duration_ms: Duration of the crossfade window in milliseconds (default: 15.0ms).

    Returns:
        Concatenated audio buffer with length len(filler) + len(response) - fade_samples.
    """
    if len(filler) == 0:
        return response.copy()
    if len(response) == 0:
        return filler.copy()

    is_int16 = (filler.dtype == np.int16 or response.dtype == np.int16)
    f_float = int16_to_float32(filler) if filler.dtype == np.int16 else filler.astype(np.float32)
    r_float = int16_to_float32(response) if response.dtype == np.int16 else response.astype(np.float32)

    fade_samples = int(round(sample_rate * (duration_ms / 1000.0)))
    fade_samples = min(fade_samples, len(f_float), len(r_float))

    if fade_samples <= 0:
        out = np.concatenate([f_float, r_float])
    elif fade_samples == 1:
        w_mid = float(1.0 / np.sqrt(2.0))
        overlap = np.array([f_float[-1] * w_mid + r_float[0] * w_mid], dtype=np.float32)
        out = np.concatenate([f_float[:-1], overlap, r_float[1:]])
    else:
        # Equal-power sine/cosine curve:
        # t runs from 0 to fade_samples - 1
        t = np.arange(fade_samples, dtype=np.float32)
        phi = (np.pi / 2.0) * (t / float(fade_samples - 1))
        cos_w = np.cos(phi)
        sin_w = np.sin(phi)

        overlap = (f_float[-fade_samples:] * cos_w) + (r_float[:fade_samples] * sin_w)
        out = np.concatenate([f_float[:-fade_samples], overlap, r_float[fade_samples:]])

    # Safeguard against accidental constructive clipping, NaNs, and Infs
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    out = np.clip(out, -1.0, 1.0)

    if is_int16:
        return float32_to_int16(out)
    return out


# Alias and compatibility wrappers
def apply_crossfade(
    audio_tail: np.ndarray,
    audio_head: np.ndarray,
    overlap_samples: int,
    curve: str = "equal_power"
) -> np.ndarray:
    """
    Stitches audio_tail and audio_head with an overlap region.
    Supports 'equal_power' (sine/cosine) and 'linear' curves.
    """
    if len(audio_tail) == 0:
        return audio_head.copy()
    if len(audio_head) == 0:
        return audio_tail.copy()

    overlap_samples = min(overlap_samples, len(audio_tail), len(audio_head))
    if overlap_samples <= 0:
        return np.concatenate([audio_tail, audio_head])

    prefix = audio_tail[:-overlap_samples]
    suffix = audio_head[overlap_samples:]
    tail_overlap = audio_tail[-overlap_samples:]
    head_overlap = audio_head[:overlap_samples]

    t = np.linspace(0.0, 1.0, overlap_samples, endpoint=True, dtype=np.float32)
    if curve == "equal_power":
        fade_out = np.cos(0.5 * np.pi * t)
        fade_in = np.sin(0.5 * np.pi * t)
    else:  # linear
        fade_out = 1.0 - t
        fade_in = t

    blended = (tail_overlap * fade_out) + (head_overlap * fade_in)
    out = np.concatenate([prefix, blended, suffix])
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(out, -1.0, 1.0)


def apply_fade_in(
    audio: np.ndarray,
    fade_samples: int,
    curve: str = "equal_power"
) -> np.ndarray:
    """Applies fade-in to the start of an audio array."""
    if len(audio) == 0 or fade_samples <= 0:
        return audio.copy()
    fade_samples = min(fade_samples, len(audio))
    out = audio.copy()
    t = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32)
    if curve == "equal_power":
        envelope = np.sin(0.5 * np.pi * t)
    else:
        envelope = t
    out[:fade_samples] *= envelope
    return out


def apply_fade_out(
    audio: np.ndarray,
    fade_samples: int,
    curve: str = "equal_power"
) -> np.ndarray:
    """Applies fade-out to the end of an audio array."""
    if len(audio) == 0 or fade_samples <= 0:
        return audio.copy()
    fade_samples = min(fade_samples, len(audio))
    out = audio.copy()
    t = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32)
    if curve == "equal_power":
        envelope = np.cos(0.5 * np.pi * t)
    else:
        envelope = 1.0 - t
    out[-fade_samples:] *= envelope
    return out


# ==============================================================================
# 5. STREAMING CROSSFADE BUFFER (REAL-TIME QUEUE)
# ==============================================================================

class CrossfadeBuffer:
    """
    Thread-safe and async-safe streaming audio ring buffer with equal-power micro-crossfading.
    Allows zero-latency handover between playing fillers and incoming LLM response audio,
    sub-50ms instant barge-in flushing, and seamless frame extraction.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        default_fade_ms: float = 15.0,
        dtype: type = np.float32,
        curve: str = "equal_power",
        default_crossfade_ms: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.default_fade_ms = default_crossfade_ms if default_crossfade_ms is not None else default_fade_ms
        self.default_crossfade_ms = self.default_fade_ms  # compatibility alias
        self.dtype = dtype
        self.curve = curve

        self._lock = threading.Lock()
        self._buffer: np.ndarray = np.zeros(0, dtype=self.dtype)
        self._is_active = True
        self._event = asyncio.Event()

    @property
    def remaining_samples(self) -> int:
        """Returns number of audio samples currently in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def available_samples(self) -> int:
        """Alias for remaining_samples."""
        return self.remaining_samples

    @property
    def remaining_duration_ms(self) -> float:
        """Returns duration of buffered audio in milliseconds."""
        with self._lock:
            return (len(self._buffer) * 1000.0) / float(self.sample_rate)

    @property
    def available_ms(self) -> float:
        """Alias for remaining_duration_ms."""
        return self.remaining_duration_ms

    @property
    def is_empty(self) -> bool:
        """Returns True if no audio is buffered."""
        with self._lock:
            return len(self._buffer) == 0

    def has_data(self) -> bool:
        """Returns True if audio is buffered."""
        return not self.is_empty

    def write(self, samples: Union[np.ndarray, bytes]) -> None:
        """
        Appends raw audio samples to the end of the buffer without crossfading.
        """
        self.write_chunk(samples, crossfade=False)

    def write_chunk(
        self,
        pcm_samples: Union[np.ndarray, bytes],
        crossfade: bool = False,
        crossfade_ms: Optional[float] = None
    ) -> None:
        """
        Appends an audio chunk to the queue. If crossfade=True and queue already has data,
        it performs an equal-power crossfade over crossfade_ms.
        """
        if isinstance(pcm_samples, (bytes, bytearray)):
            int16_arr = np.frombuffer(pcm_samples, dtype=np.int16)
            if self.dtype == np.float32:
                samples = int16_to_float32(int16_arr)
            else:
                samples = int16_arr
        else:
            samples = np.asarray(pcm_samples)
            if samples.dtype != self.dtype:
                if self.dtype == np.float32:
                    samples = int16_to_float32(samples)
                else:
                    samples = float32_to_int16(samples)

        if len(samples) == 0:
            return

        with self._lock:
            if not self._is_active:
                return

            if not crossfade or len(self._buffer) == 0:
                self._buffer = np.concatenate([self._buffer, samples])
            else:
                cf_ms = crossfade_ms or self.default_fade_ms
                self._buffer = equal_power_crossfade(
                    self._buffer,
                    samples,
                    sample_rate=self.sample_rate,
                    duration_ms=cf_ms
                )

        self._event.set()

    def crossfade_append(
        self,
        next_samples: Union[np.ndarray, bytes],
        duration_ms: Optional[float] = None
    ) -> None:
        """
        Appends next_samples to the buffer by performing an equal-power crossfade
        with the tail of the currently buffered audio.
        """
        self.write_chunk(next_samples, crossfade=True, crossfade_ms=duration_ms)

    def crossfade_cut_in(
        self,
        new_samples: Union[np.ndarray, bytes],
        duration_ms: Optional[float] = None
    ) -> None:
        """
        Barge-in / Early Filler Termination Handover:
        Cuts into whatever audio is currently buffered! Crossfades the immediate next
        duration_ms of remaining buffer into the start of new_samples, discarding the
        remainder of the old audio (e.g. cutting off remaining filler without clicks or latency).
        """
        if isinstance(new_samples, (bytes, bytearray)):
            int16_arr = np.frombuffer(new_samples, dtype=np.int16)
            samples = int16_to_float32(int16_arr) if self.dtype == np.float32 else int16_arr
        else:
            samples = np.asarray(new_samples)
            if samples.dtype != self.dtype:
                samples = int16_to_float32(samples) if self.dtype == np.float32 else float32_to_int16(samples)

        if len(samples) == 0:
            return

        fade_ms = duration_ms or self.default_fade_ms
        fade_samples = int(round(self.sample_rate * (fade_ms / 1000.0)))

        with self._lock:
            if not self._is_active:
                return

            if len(self._buffer) == 0 or fade_samples <= 0:
                self._buffer = samples.copy()
            else:
                head_len = min(len(self._buffer), fade_samples)
                transition_head = self._buffer[:head_len]
                self._buffer = equal_power_crossfade(
                    transition_head,
                    samples,
                    sample_rate=self.sample_rate,
                    duration_ms=fade_ms
                )

        self._event.set()

    def read(self, num_samples: int) -> np.ndarray:
        """
        Pops up to num_samples from the head of the buffer synchronously.
        """
        return self.read_samples(num_samples)

    def read_samples(self, num_samples: int) -> np.ndarray:
        """
        Pops up to num_samples from the head of the buffer synchronously.
        If fewer samples are available, returns whatever is remaining.
        """
        with self._lock:
            if len(self._buffer) == 0:
                self._event.clear()
                return np.zeros(0, dtype=self.dtype)

            take = min(num_samples, len(self._buffer))
            chunk = self._buffer[:take].copy()
            self._buffer = self._buffer[take:]

            if len(self._buffer) == 0:
                self._event.clear()

            return chunk

    def read_frame(self, frame_ms: float = 20.0, pad: bool = False) -> np.ndarray:
        """
        Reads a fixed-duration frame (e.g. 20ms = 480 samples at 24kHz, 320 at 16kHz).
        If pad is True and buffer has fewer samples, pads with silence to frame size.
        """
        target_samples = int(round(self.sample_rate * (frame_ms / 1000.0)))
        chunk = self.read_samples(target_samples)
        if pad and len(chunk) < target_samples:
            padding = np.zeros(target_samples - len(chunk), dtype=self.dtype)
            chunk = np.concatenate([chunk, padding])
        return chunk

    async def read_frame_async(
        self,
        num_samples: int,
        timeout: Optional[float] = None
    ) -> np.ndarray:
        """
        Async wait until at least some samples are available and pop them.
        """
        while True:
            with self._lock:
                if len(self._buffer) > 0:
                    take = min(num_samples, len(self._buffer))
                    chunk = self._buffer[:take].copy()
                    self._buffer = self._buffer[take:]
                    if len(self._buffer) == 0:
                        self._event.clear()
                    return chunk

            try:
                await asyncio.wait_for(self._event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return np.zeros(0, dtype=self.dtype)

    def peek(self, num_samples: int) -> np.ndarray:
        """
        Views the next num_samples without removing them from the buffer.
        """
        with self._lock:
            take = min(num_samples, len(self._buffer))
            return self._buffer[:take].copy()

    def flush(self) -> None:
        """
        Instantly clears all queued audio for sub-50ms barge-in cutoff.
        """
        with self._lock:
            self._buffer = np.zeros(0, dtype=self.dtype)
            self._event.clear()

    def clear(self) -> None:
        """Alias for flush()."""
        self.flush()

    def reset(self) -> None:
        """Resets buffer state."""
        self.flush()
        self._is_active = True
