"""
Echo Canceller & Inbound Audio Processor for Ultra-Low Latency Telephony.
Implements:
1. EchoReferenceTap: Circular ring buffer storing outbound synthesized audio frames.
2. InboundAudioProcessor / EchoCanceller:
   - Rolling reference of outbound audio with estimated round-trip delay (40ms-120ms).
   - Energy subtraction and cross-correlation estimation to cancel acoustic echo on speakerphones.
   - Distinguishes between backchanneling sounds ("uh-huh", "yeah", "mhm" < 350ms, low energy)
     vs true barge-in / interruption ("wait stop", "no", continuous speech > 300ms).
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

    @property
    def total_samples_written(self) -> int:
        return self._total_samples_written

    @property
    def last_push_time(self) -> float:
        return self._last_push_time

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

    def push_reference(
        self,
        pcm_data: Union[np.ndarray, bytes],
        sample_rate: Optional[int] = None
    ) -> None:
        """
        Pushes an outbound audio chunk into the circular buffer.
        Automatically converts to float32 mono at self.sample_rate.
        """
        in_rate = sample_rate or self.sample_rate
        samples = ensure_float32(pcm_data, source_rate=in_rate, target_rate=self.sample_rate)
        n = len(samples)
        if n == 0:
            return

        with self._lock:
            self._last_push_time = time.time()
            self._active_outbound = True

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

    def get_delayed_reference(
        self,
        num_samples: int,
        delay_ms: float = 60.0
    ) -> np.ndarray:
        """
        Retrieves reference audio slice aligned with propagation delay.
        """
        delay_samples = int(round(delay_ms * self.sample_rate / 1000.0))
        num_samples = min(num_samples, self.buffer_size - delay_samples)
        if num_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        with self._lock:
            if self._total_samples_written < (delay_samples + num_samples):
                # Not enough history written yet
                return np.zeros(num_samples, dtype=np.float32)

            start_idx = self._write_pos - delay_samples - num_samples
            idx = np.arange(start_idx, start_idx + num_samples) % self.buffer_size
            return self._buffer[idx].copy()

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
    - Estimates round-trip propagation delay (40ms-120ms typical for PSTN/WebRTC).
    - Energy subtraction and cross-correlation estimation to cancel speakerphone acoustic echo.
    - Distinguishes between backchanneling sounds ("uh-huh", "yeah", "mhm" < 350ms, low energy)
      vs true barge-in / interruption ("wait stop", "no", continuous speech > 300ms).
    - Exposes should_interrupt(inbound_pcm) -> bool.
    """

    def __init__(
        self,
        echo_tap: Optional[EchoReferenceTap] = None,
        sample_rate: int = 16000,
        min_delay_ms: float = 40.0,
        max_delay_ms: float = 120.0,
        initial_delay_ms: float = 65.0,
        backchannel_max_duration_ms: float = 350.0,
        backchannel_max_energy: float = 0.075,
        barge_in_min_duration_ms: float = 300.0,
        speech_energy_threshold: float = 0.025,
        barge_in_urgent_energy: float = 0.16,
        echo_correlation_threshold: float = 0.40,
        echo_subtraction_factor: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.echo_tap = echo_tap or EchoReferenceTap(sample_rate=sample_rate)
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.estimated_delay_ms = initial_delay_ms

        # Thresholds
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
        self._last_process_time = time.time()

        # Rolling delay alignment history
        self._delay_history: deque[float] = deque(maxlen=20)
        self._delay_history.append(initial_delay_ms)

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

        # Retrieve reference window covering the max possible delay + chunk length
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

        # Sweep candidate delays with a step of 4 samples (~0.25ms at 16kHz) for efficiency
        step = 4
        best_corr = -1.0
        best_lag = min_lag

        for lag in range(min_lag, max_lag, step):
            # The reference slice that would have arrived after propagation lag
            ref_slice = ref_window[lag:lag + num_samples]
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
        best_slice = ref_window[best_lag:best_lag + num_samples]

        # Update rolling smoothed delay if correlation is confident
        if best_corr > 0.45:
            self._delay_history.append(best_delay_ms)
            self.estimated_delay_ms = float(np.median(self._delay_history))

        return max(0.0, best_corr), best_delay_ms, best_slice

    def cancel_echo(
        self,
        inbound: np.ndarray,
        ref_slice: np.ndarray,
        correlation: float
    ) -> Tuple[np.ndarray, bool]:
        """
        Applies energy subtraction and echo attenuation based on cross-correlation.
        Returns (cleaned_audio, is_echo).
        """
        in_rms = float(np.sqrt(np.mean(inbound ** 2)))
        ref_rms = float(np.sqrt(np.mean(ref_slice ** 2))) if len(ref_slice) == len(inbound) else 0.0

        if in_rms < 1e-4 or ref_rms < 1e-4:
            return inbound, False

        # If correlation is high, incoming sound is overwhelmingly acoustic echo
        is_echo = correlation >= self.echo_correlation_threshold

        if not is_echo:
            # Low correlation: return inbound with no attenuation
            return inbound, False

        # Compute scaling factor alpha via projection: alpha = cov(inbound, ref) / var(ref)
        ref_var = float(np.var(ref_slice))
        if ref_var > 1e-5:
            alpha = float(np.dot(inbound, ref_slice) / (len(inbound) * ref_var))
            alpha = max(0.0, min(alpha * self.echo_subtraction_factor, 1.5))
        else:
            alpha = correlation * (in_rms / ref_rms)

        # Time-domain subtraction
        echo_est = alpha * ref_slice
        subtracted = inbound - echo_est

        # Residual energy calculation
        sub_rms = float(np.sqrt(np.mean(subtracted ** 2)))

        # Suppression gain: suppress aggressively if correlation is very high (>0.6)
        if correlation > 0.65:
            suppression_gain = 0.05
        elif correlation > 0.50:
            suppression_gain = 0.15
        else:
            suppression_gain = max(0.2, 1.0 - correlation)

        cleaned = subtracted * suppression_gain
        return cleaned, True

    def process_frame(
        self,
        inbound_pcm: Union[np.ndarray, bytes],
        sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, bool, ProcessingMetrics]:
        """
        Processes an inbound audio chunk:
        - Resamples to 16kHz float32.
        - Calculates echo correlation and cancels acoustic echo.
        - Classifies backchannel vs true barge-in interruption.
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
        # If echo cancellation determined this was echo, speech power is suppressed
        is_speech = (cleaned_rms >= self.speech_energy_threshold) and not is_echo

        is_backchannel = False
        is_barge_in = False

        if bot_is_speaking:
            if is_speech:
                self._speech_duration_ms += frame_duration_ms
                self._silence_duration_ms = 0.0
                self._burst_frame_count += 1
                self._burst_avg_energy_sum += cleaned_rms
                if cleaned_rms > self._burst_peak_energy:
                    self._burst_peak_energy = cleaned_rms

                avg_energy = self._burst_avg_energy_sum / max(1, self._burst_frame_count)

                # Condition 1: Continuous speech exceeding barge_in_min_duration_ms (>300ms)
                if self._speech_duration_ms >= self.barge_in_min_duration_ms:
                    is_barge_in = True
                    self._interrupted = True

                # Condition 2: Urgent loud barge-in ("STOP!", "NO!", "WAIT!")
                elif self._speech_duration_ms >= 140.0 and self._burst_peak_energy >= self.barge_in_urgent_energy:
                    is_barge_in = True
                    self._interrupted = True

            else:
                self._silence_duration_ms += frame_duration_ms

                # If speech was present and has now fallen silent:
                if self._silence_duration_ms >= 150.0 and self._speech_duration_ms > 0:
                    avg_energy = self._burst_avg_energy_sum / max(1, self._burst_frame_count)

                    # Backchannel check: brief duration (< 350ms) and low-to-moderate energy
                    if (self._speech_duration_ms < self.backchannel_max_duration_ms and
                            avg_energy <= self.backchannel_max_energy and
                            not self._interrupted):
                        is_backchannel = True

                    # Reset speech tracking for next burst
                    self._speech_duration_ms = 0.0
                    self._burst_frame_count = 0
                    self._burst_avg_energy_sum = 0.0
                    self._burst_peak_energy = 0.0
                    self._interrupted = False
        else:
            # Bot is NOT speaking: reset barge-in tracking
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
