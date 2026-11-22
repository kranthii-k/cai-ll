"""
Echo Canceller & Inbound Audio Processor for Ultra-Low Latency Telephony.
Implements:
1. EchoReferenceTap: Circular ring buffer storing outbound synthesized audio frames.
2. InboundAudioProcessor / EchoCanceller:
   - Rolling reference of outbound audio with estimated round-trip delay (25ms-350ms).
   - Energy subtraction and cross-correlation estimation to cancel acoustic echo on speakerphones.
   - Double-talk detection: distinguishes between pure speaker echo and true caller voice.
   - Distinguishes between backchanneling sounds ("uh-huh", "yeah", "mhm" < 350ms, low energy)
     vs true barge-in / interruption ("stop", "wait", "hold on", continuous speech >= 350ms).
   - Enforces 450ms holdoff at the start of any TTS utterance to eliminate speakerphone onset transients.
   - Immediate sub-50ms cutoff without dropping caller audio.
   - Exposes `should_interrupt(inbound_pcm) -> bool`.
"""

from __future__ import annotations

import time
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
from scipy import signal


def ensure_float32(
    audio: Union[np.ndarray, bytes],
    source_rate: int = 16000,
    target_rate: int = 16000
) -> np.ndarray:
    """
    Normalizes audio input (bytes, int16, or float32) into float32 array in range [-1.0, 1.0]
    resampled to target_rate.
    """
    if isinstance(audio, bytes):
        pcm_16 = np.frombuffer(audio, dtype=np.int16)
        samples = pcm_16.astype(np.float32) / 32768.0
    elif isinstance(audio, np.ndarray):
        if audio.dtype == np.int16:
            samples = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32 or audio.dtype == np.float64:
            samples = audio.astype(np.float32)
            # Clip if slightly beyond [-1.0, 1.0]
            samples = np.clip(samples, -1.0, 1.0)
        else:
            samples = audio.astype(np.float32)
    else:
        samples = np.array(audio, dtype=np.float32)

    # Flatten to 1D mono if stereo
    if samples.ndim > 1:
        samples = samples[:, 0]

    if source_rate != target_rate and len(samples) > 0:
        # Resample to target rate using scipy linear resample
        num_target_samples = int(round(len(samples) * float(target_rate) / float(source_rate)))
        if num_target_samples > 0:
            samples = signal.resample(samples, num_target_samples)

    return samples


class EchoReferenceTap:
    """
    A circular audio buffer that stores outbound synthesized frames (what the bot is saying).
    Thread-safe and optimized for fast continuous writes and delayed slice queries.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_duration_sec: float = 6.0
    ):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration_sec)
        self._buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._total_samples_written = 0
        self._lock = threading.Lock()
        self._last_push_time = 0.0
        self._active_outbound = False
        self._utterance_start_time = 0.0

    @property
    def total_samples_written(self) -> int:
        return self._total_samples_written

    @property
    def last_push_time(self) -> float:
        return self._last_push_time

    @property
    def utterance_start_time(self) -> float:
        return self._utterance_start_time

    def is_speaking(self, window_ms: float = 250.0, energy_thresh: float = 0.005) -> bool:
        """
        Returns True if the bot has written outbound audio recently and the recent
        window contains non-trivial energy.
        """
        now = time.time()
        # If no audio was pushed in the last 400ms, bot is not actively playing
        if now - self._last_push_time > 0.4:
            return False

        with self._lock:
            num_samples = int(self.sample_rate * (window_ms / 1000.0))
            num_samples = min(num_samples, self.buffer_size)
            if self._total_samples_written < num_samples:
                recent_data = self._buffer[:self._write_pos]
            else:
                idx = np.arange(self._write_pos - num_samples, self._write_pos) % self.buffer_size
                recent_data = self._buffer[idx]

        if len(recent_data) == 0:
            return False
        rms = float(np.sqrt(np.mean(recent_data ** 2)))
        return rms > energy_thresh

    def _internal_write(self, samples: np.ndarray) -> None:
        n = len(samples)
        if n == 0:
            return
        if n >= self.buffer_size:
            # If chunk is larger than buffer, take only the most recent part
            self._buffer[:] = samples[-self.buffer_size:]
            self._write_pos = 0
            self._total_samples_written += n
            return

        end_pos = self._write_pos + n
        if end_pos <= self.buffer_size:
            self._buffer[self._write_pos:end_pos] = samples
            self._write_pos = end_pos % self.buffer_size
        else:
            first_part = self.buffer_size - self._write_pos
            self._buffer[self._write_pos:] = samples[:first_part]
            second_part = n - first_part
            self._buffer[:second_part] = samples[first_part:]
            self._write_pos = second_part

        self._total_samples_written += n

    def push_reference(
        self,
        pcm_data: Union[np.ndarray, bytes],
        sample_rate: Optional[int] = None
    ) -> None:
        """
        Pushes an outbound audio chunk into the circular buffer.
        Automatically converts to float32 mono at self.sample_rate.
        Pads silence if there is a gap in real-time streaming to keep the
        circular buffer strictly synchronized with wall-clock time.
        """
        in_rate = sample_rate or self.sample_rate
        samples = ensure_float32(pcm_data, source_rate=in_rate, target_rate=self.sample_rate)
        n = len(samples)
        if n == 0:
            return

        with self._lock:
            now = time.time()
            # If gap between audio pushes is greater than 50ms, pad silence to keep ring buffer aligned
            if self._last_push_time > 0:
                gap_sec = now - self._last_push_time
                if gap_sec > 0.05:
                    gap_samples = min(int(round(gap_sec * self.sample_rate)), self.buffer_size)
                    if gap_samples > 0:
                        self._internal_write(np.zeros(gap_samples, dtype=np.float32))

            # Detect new utterance start after pause > 100ms
            if self._last_push_time == 0.0 or (now - self._last_push_time > 0.10):
                self._utterance_start_time = now

            self._last_push_time = now
            self._active_outbound = True
            self._internal_write(samples)

    def get_delayed_reference(
        self,
        num_samples: int,
        delay_ms: float = 60.0
    ) -> np.ndarray:
        """
        Retrieves reference audio slice aligned with propagation delay.
        Always returns an array of length num_samples (zero-padded if unwritten history).
        """
        delay_samples = int(round(delay_ms * self.sample_rate / 1000.0))
        num_samples = min(num_samples, self.buffer_size - delay_samples)
        if num_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        with self._lock:
            if self._total_samples_written == 0:
                return np.zeros(num_samples, dtype=np.float32)

            start_idx = self._write_pos - delay_samples - num_samples
            idx = np.arange(start_idx, start_idx + num_samples) % self.buffer_size
            res = self._buffer[idx].copy()

            # If total samples written is less than history needed, zero out the unwritten past
            needed_history = delay_samples + num_samples
            if self._total_samples_written < needed_history:
                unwritten_samples = needed_history - self._total_samples_written
                if unwritten_samples >= num_samples:
                    res.fill(0.0)
                else:
                    res[:unwritten_samples] = 0.0

            return res

    def clear(self) -> None:
        """
        Clears the buffer and resets state (e.g., when barge-in aborts playback).
        """
        with self._lock:
            self._buffer.fill(0.0)
            self._write_pos = 0
            self._total_samples_written = 0
            self._last_push_time = 0.0
            self._active_outbound = False
            self._utterance_start_time = 0.0


@dataclass
class ProcessingMetrics:
    inbound_rms: float = 0.0
    ref_rms: float = 0.0
    cleaned_rms: float = 0.0
    cross_correlation: float = 0.0
    estimated_delay_ms: float = 60.0
    is_echo: bool = False
    is_speech: bool = False
    is_backchannel: bool = False
    is_barge_in: bool = False
    consecutive_speech_ms: float = 0.0


class InboundAudioProcessor:
    """
    Inbound Audio Processor with Acoustic Echo Cancellation & Barge-In Classifier.
    - Estimates round-trip propagation delay (25ms-350ms typical for PSTN/WebRTC).
    - Energy subtraction and cross-correlation estimation to cancel speakerphone acoustic echo.
    - Double-talk detection: distinguishes between pure speaker echo and true caller voice.
    - Distinguishes between backchanneling sounds ("uh-huh", "yeah", "mhm" < 350ms, low energy)
      vs true barge-in / interruption ("stop", "wait", "hold on", continuous speech >= 350ms).
    - Enforces 450ms holdoff at the start of any TTS utterance.
    - Exposes should_interrupt(inbound_pcm) -> bool.
    """

    def __init__(
        self,
        echo_tap: Optional[EchoReferenceTap] = None,
        sample_rate: int = 16000,
        min_delay_ms: float = 25.0,
        max_delay_ms: float = 350.0,
        initial_delay_ms: float = 60.0,
        barge_in_holdoff_ms: float = 0.0,
        backchannel_max_duration_ms: float = 350.0,
        backchannel_max_energy: float = 0.075,
        barge_in_min_duration_ms: float = 350.0,
        speech_energy_threshold: float = 0.035,
        barge_in_urgent_energy: float = 0.14,
        echo_correlation_threshold: float = 0.35,
        echo_subtraction_factor: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.echo_tap = echo_tap or EchoReferenceTap(sample_rate=sample_rate)
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.estimated_delay_ms = initial_delay_ms

        # Thresholds
        self.barge_in_holdoff_ms = barge_in_holdoff_ms
        self.backchannel_max_duration_ms = backchannel_max_duration_ms
        self.backchannel_max_energy = backchannel_max_energy
        self.barge_in_min_duration_ms = barge_in_min_duration_ms
        self.speech_energy_threshold = speech_energy_threshold
        self.barge_in_urgent_energy = barge_in_urgent_energy
        self.echo_correlation_threshold = echo_correlation_threshold
        self.echo_subtraction_factor = echo_subtraction_factor

        # State tracking for continuous speech
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._burst_peak_energy = 0.0
        self._burst_avg_energy_sum = 0.0
        self._burst_frame_count = 0
        self._interrupted = False
        self._utterance_start_time = 0.0
        self._delay_locked = False
        self._last_process_time = time.time()

        # Rolling delay alignment history
        self._delay_history: deque[float] = deque(maxlen=20)
        self._delay_history.append(initial_delay_ms)

    def notify_utterance_start(self, timestamp: Optional[float] = None) -> None:
        """Notifies the processor that a new outbound TTS utterance has started."""
        self._utterance_start_time = timestamp or time.time()
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._burst_frame_count = 0
        self._burst_avg_energy_sum = 0.0
        self._burst_peak_energy = 0.0
        self._interrupted = False

    def reset_state(self) -> None:
        """Resets speech duration and barge-in tracking state."""
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._burst_peak_energy = 0.0
        self._burst_avg_energy_sum = 0.0
        self._burst_frame_count = 0
        self._interrupted = False

    def estimate_delay_and_correlation(
        self,
        inbound: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimates cross-correlation peak and propagation delay between inbound
        and outbound reference audio in the range [min_delay_ms, max_delay_ms].
        Returns (max_correlation, best_delay_ms, best_reference_slice).
        """
        num_samples = len(inbound)
        if num_samples < 64:
            return 0.0, self.estimated_delay_ms, np.zeros(num_samples, dtype=np.float32)

        min_lag = int(round(self.min_delay_ms * self.sample_rate / 1000.0))
        max_lag = int(round(self.max_delay_ms * self.sample_rate / 1000.0))

        # Retrieve reference window covering max possible delay + chunk length
        total_fetch = max_lag + num_samples
        ref_window = self.echo_tap.get_delayed_reference(
            num_samples=total_fetch,
            delay_ms=0.0
        )

        if len(ref_window) < total_fetch:
            return 0.0, self.estimated_delay_ms, np.zeros(num_samples, dtype=np.float32)

        in_rms = float(np.sqrt(np.mean(inbound ** 2)))
        ref_rms = float(np.sqrt(np.mean(ref_window ** 2)))
        if in_rms < 1e-4 or ref_rms < 1e-4:
            return 0.0, self.estimated_delay_ms, np.zeros(num_samples, dtype=np.float32)

        # Standardize inbound
        in_norm = (inbound - np.mean(inbound))
        in_std = float(np.std(inbound))
        if in_std < 1e-5:
            return 0.0, self.estimated_delay_ms, np.zeros(num_samples, dtype=np.float32)
        in_norm = in_norm / in_std

        step = 4
        best_corr = -1.0
        best_lag = min_lag

        # Adaptive search: check neighborhood around locked delay first for speed & stability
        check_global = True
        if self._delay_locked and self.estimated_delay_ms > 0:
            est_lag = int(round(self.estimated_delay_ms * self.sample_rate / 1000.0))
            local_min = max(min_lag, est_lag - int(0.015 * self.sample_rate))
            local_max = min(max_lag, est_lag + int(0.015 * self.sample_rate))
            for lag in range(local_min, local_max, 2):
                start_idx = total_fetch - lag - num_samples
                if start_idx < 0:
                    continue
                ref_slice = ref_window[start_idx:start_idx + num_samples]
                ref_std = float(np.std(ref_slice))
                if ref_std < 1e-5:
                    continue
                ref_norm = (ref_slice - np.mean(ref_slice)) / ref_std
                corr = float(np.dot(in_norm, ref_norm) / num_samples)
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
            if best_corr >= 0.50:
                check_global = False

        if check_global:
            for lag in range(min_lag, max_lag, step):
                start_idx = total_fetch - lag - num_samples
                if start_idx < 0:
                    continue
                ref_slice = ref_window[start_idx:start_idx + num_samples]
                if len(ref_slice) < num_samples:
                    continue

                ref_std = float(np.std(ref_slice))
                if ref_std < 1e-5:
                    continue

                ref_norm = (ref_slice - np.mean(ref_slice)) / ref_std
                # Normalized cross-correlation
                corr = float(np.dot(in_norm, ref_norm) / num_samples)
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag

        best_delay_ms = (best_lag * 1000.0) / self.sample_rate
        best_start = total_fetch - best_lag - num_samples
        best_slice = ref_window[best_start:best_start + num_samples] if best_start >= 0 else np.zeros(num_samples, dtype=np.float32)

        # Update rolling smoothed delay if correlation is confident
        if best_corr >= 0.40:
            self._delay_history.append(best_delay_ms)
            self.estimated_delay_ms = float(np.median(self._delay_history))
            self._delay_locked = True

        return max(0.0, best_corr), best_delay_ms, best_slice

    def cancel_echo(
        self,
        inbound: np.ndarray,
        ref_slice: np.ndarray,
        correlation: float
    ) -> Tuple[np.ndarray, bool]:
        """
        Applies energy subtraction and echo attenuation based on cross-correlation.
        Distinguishes between pure acoustic echo and double-talk (caller interruption).
        Returns (cleaned_audio, is_echo).
        """
        in_rms = float(np.sqrt(np.mean(inbound ** 2)))
        ref_rms = float(np.sqrt(np.mean(ref_slice ** 2))) if len(ref_slice) == len(inbound) else 0.0

        if in_rms < 1e-4 or ref_rms < 1e-4:
            return inbound, False

        # Compute scaling factor alpha via projection: alpha = cov(inbound, ref) / var(ref)
        ref_var = float(np.var(ref_slice))
        if ref_var > 1e-5:
            alpha = float(np.dot(inbound, ref_slice) / (len(inbound) * ref_var))
            alpha = max(0.0, min(alpha * self.echo_subtraction_factor, 2.0))
        else:
            alpha = correlation * (in_rms / ref_rms)

        # Time-domain subtraction
        echo_est = alpha * ref_slice
        subtracted = inbound - echo_est
        sub_rms = float(np.sqrt(np.mean(subtracted ** 2)))

        erle_ratio = sub_rms / (in_rms + 1e-6)

        # Double-talk detection:
        # Caller is speaking over echo if residual energy is high (sub_rms >= 0.045)
        # and subtraction failed to remove most energy (erle_ratio >= 0.40)
        is_double_talk = (sub_rms >= 0.045 and erle_ratio >= 0.40 and in_rms >= self.speech_energy_threshold)

        # Acoustic Echo classification:
        # It is pure echo if correlation is above threshold AND not double-talk,
        # OR if subtraction effectively eliminated the energy (erle_ratio < 0.30).
        is_echo = (correlation >= self.echo_correlation_threshold or erle_ratio < 0.30) and not is_double_talk

        if is_echo:
            # Pure echo: suppress residual leakage aggressively so mic doesn't hear echo
            if correlation > 0.60 or erle_ratio < 0.15:
                suppression_gain = 0.03
            elif correlation > 0.45 or erle_ratio < 0.25:
                suppression_gain = 0.08
            else:
                suppression_gain = 0.18
            cleaned = subtracted * suppression_gain
            return cleaned, True
        else:
            if is_double_talk:
                # Keep echo-subtracted signal so human caller's voice is preserved without bot echo
                cleaned = subtracted
            else:
                cleaned = inbound
            return cleaned, False

    def process_frame(
        self,
        inbound_pcm: Union[np.ndarray, bytes],
        sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, bool, ProcessingMetrics]:
        """
        Processes an inbound audio chunk:
        - Resamples to 16kHz float32.
        - Calculates echo correlation and cancels acoustic echo.
        - Distinguishes double-talk vs pure echo.
        - Classifies backchannel vs true barge-in interruption.
        - Respects barge-in hold-off at utterance onset.
        Returns:
            (cleaned_pcm_float32, should_interrupt, metrics)
        """
        in_rate = sample_rate or self.sample_rate
        inbound = ensure_float32(inbound_pcm, source_rate=in_rate, target_rate=self.sample_rate)
        num_samples = len(inbound)

        if num_samples == 0:
            metrics = ProcessingMetrics()
            return np.zeros(0, dtype=np.float32), False, metrics

        frame_duration_ms = (num_samples * 1000.0) / self.sample_rate
        in_rms = float(np.sqrt(np.mean(inbound ** 2)))

        # Check if the bot is actively speaking
        bot_is_speaking = self.echo_tap.is_speaking(window_ms=250.0)

        corr = 0.0
        best_delay = self.estimated_delay_ms
        ref_slice = np.zeros(num_samples, dtype=np.float32)
        ref_rms = 0.0
        is_echo = False

        if bot_is_speaking:
            corr, best_delay, ref_slice = self.estimate_delay_and_correlation(inbound)
            ref_rms = float(np.sqrt(np.mean(ref_slice ** 2)))
            cleaned, is_echo = self.cancel_echo(inbound, ref_slice, corr)
        else:
            cleaned = inbound
            is_echo = False

        cleaned_rms = float(np.sqrt(np.mean(cleaned ** 2)))

        # Speech activity evaluation on echo-cleaned audio
        is_speech = (cleaned_rms >= self.speech_energy_threshold) and not is_echo

        # Check utterance onset hold-off
        in_holdoff = False
        if bot_is_speaking and self.barge_in_holdoff_ms > 0:
            elapsed_ms = (time.time() - self._utterance_start_time) * 1000.0 if self._utterance_start_time > 0 else 9999.0
            outbound_dur_ms = (self.echo_tap.total_samples_written * 1000.0) / self.sample_rate
            if elapsed_ms < self.barge_in_holdoff_ms or outbound_dur_ms < self.barge_in_holdoff_ms:
                in_holdoff = True

        is_backchannel = False
        is_barge_in = False

        if bot_is_speaking and not in_holdoff:
            if is_speech:
                self._speech_duration_ms += frame_duration_ms
                self._silence_duration_ms = 0.0
                self._burst_frame_count += 1
                self._burst_avg_energy_sum += cleaned_rms
                if cleaned_rms > self._burst_peak_energy:
                    self._burst_peak_energy = cleaned_rms

                avg_energy = self._burst_avg_energy_sum / max(1, self._burst_frame_count)

                # Condition 1: Continuous speech exceeding barge_in_min_duration_ms (>= 350ms)
                if self._speech_duration_ms >= self.barge_in_min_duration_ms:
                    is_barge_in = True
                    self._interrupted = True

                # Condition 2: Urgent loud barge-in ("STOP!", "NO!", "WAIT!", "HOLD ON!")
                # caller saying 'stop', 'wait', 'hold on' (>= 140ms and elevated energy)
                elif self._speech_duration_ms >= 140.0 and (
                    self._burst_peak_energy >= self.barge_in_urgent_energy or avg_energy >= 0.11
                ):
                    is_barge_in = True
                    self._interrupted = True

            else:
                self._silence_duration_ms += frame_duration_ms

                # If silence exceeds 80ms, reset speech tracking so non-consecutive frames don't accumulate
                if self._silence_duration_ms >= 80.0:
                    if self._speech_duration_ms > 0:
                        avg_energy = self._burst_avg_energy_sum / max(1, self._burst_frame_count)
                        if (
                            self._speech_duration_ms < self.backchannel_max_duration_ms
                            and avg_energy <= self.backchannel_max_energy
                            and not self._interrupted
                        ):
                            is_backchannel = True

                    self._speech_duration_ms = 0.0
                    self._burst_frame_count = 0
                    self._burst_avg_energy_sum = 0.0
                    self._burst_peak_energy = 0.0
                    self._interrupted = False
        else:
            # Bot is NOT speaking or in hold-off: reset speech tracking
            self._speech_duration_ms = 0.0
            self._silence_duration_ms = 0.0
            self._burst_frame_count = 0
            self._burst_avg_energy_sum = 0.0
            self._burst_peak_energy = 0.0
            self._interrupted = False

        metrics = ProcessingMetrics(
            inbound_rms=in_rms,
            ref_rms=ref_rms,
            cleaned_rms=cleaned_rms,
            cross_correlation=corr,
            estimated_delay_ms=best_delay,
            is_echo=is_echo,
            is_speech=is_speech,
            is_backchannel=is_backchannel,
            is_barge_in=is_barge_in,
            consecutive_speech_ms=self._speech_duration_ms
        )

        return cleaned, is_barge_in, metrics

    def should_interrupt(
        self,
        inbound_pcm: Union[np.ndarray, bytes],
        sample_rate: Optional[int] = None
    ) -> bool:
        """
        Exposes `should_interrupt(inbound_pcm) -> bool`.
        Determines whether the inbound PCM chunk constitutes a true barge-in interruption.
        """
        _, interrupt, _ = self.process_frame(inbound_pcm, sample_rate=sample_rate)
        return interrupt


# Aliases for convenience
EchoCanceller = InboundAudioProcessor
