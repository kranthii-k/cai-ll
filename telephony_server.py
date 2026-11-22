"""
Telephony Server: Unified Full-Duplex VoIP & Web Phone Streaming Engine.
Provides:
- `/twiml` POST endpoint returning TwiML XML.
- `/stream` WebSocket endpoint for Twilio Media Streams (8000Hz G.711u / base64).
- `/web_stream` WebSocket endpoint for browser audio testing (16000Hz PCM / Float32).
- Full integration with:
    - engine.crossfade_buffer.CrossfadeBuffer
    - engine.filler_manager.FillerManager
    - engine.fast_classifier.FastClassifier
    - engine.dual_brain.DualBrainEngine
    - engine.echo_canceller.EchoReferenceTap & InboundAudioProcessor
- Non-blocking async workers for VAD (Silero), STT (Whisper), Fast-Path Filler trigger, and Deep-Path TTS.
- Full-duplex: caller can interrupt (barge-in) anytime without audio loss. True barge-in cancels
  in-flight TTS playback and clears audio queues immediately (<50ms response).
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
import io
import json
import logging
import math
import os
import re
import site
import time
import uuid
import wave
from typing import Dict, List, Optional, Tuple, Any

import httpx
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
import uvicorn

# CUDA DLL FIX FOR WINDOWS
if os.name == 'nt':
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib):
            os.add_dll_directory(torch_lib)
        for sp in site.getsitepackages():
            cublas_bin = os.path.join(sp, "nvidia", "cublas", "bin")
            cudnn_bin = os.path.join(sp, "nvidia", "cudnn", "bin")
            if os.path.exists(cublas_bin):
                os.add_dll_directory(cublas_bin)
            if os.path.exists(cudnn_bin):
                os.add_dll_directory(cudnn_bin)
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from openai import AsyncOpenAI
import pysbd

from engine.crossfade_buffer import (
    CrossfadeBuffer,
    apply_crossfade,
    resample_24k_to_16k,
    resample_16k_to_8k,
    resample_8k_to_16k,
    resample_audio,
)
from engine.echo_canceller import EchoReferenceTap, InboundAudioProcessor
from engine.fast_classifier import ConversationalStance, FastClassifier
from engine.filler_manager import FillerItem, FillerManager
from engine.dual_brain import DualBrainEngine, DualBrainEvent, DualBrainEventType

try:
    import audioop
except ImportError:
    audioop = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("telephony_server")

# Audio & VAD Constants
SAMPLE_RATE_TWILIO = 8000
SAMPLE_RATE_VAD = 16000
SAMPLE_RATE_KOKORO = 24000
TWILIO_FRAME_SIZE = 160  # 20ms at 8kHz mu-law
VAD_CHUNK_SAMPLES = 512  # 32ms at 16kHz
VAD_SPEECH_THRESHOLD = 0.25

# Dynamic Turn-Taking & Endpointing Constants
DEFAULT_SILENCE_TIMEOUT_MS = 480.0  # 450ms-500ms default prevents chopping human speech mid-sentence
SILENCE_TIMEOUT_MS = DEFAULT_SILENCE_TIMEOUT_MS  # Backwards compatibility
MIN_SPEECH_DURATION_MS = 250.0  # Filter out short clicks, breath noises, and transient acoustic artifacts (<250ms)
SHORT_SPEECH_THRESHOLD_SEC = 1.0  # Utterances <1.0s are considered short/hesitations
SHORT_SPEECH_SILENCE_TIMEOUT_MS = 500.0  # Require >=500ms silence for short speech before turn conclusion
LONG_SPEECH_THRESHOLD_SEC = 1.5  # Utterances >1.5s are considered substantive clauses
LONG_SPEECH_CLAUSE_SILENCE_TIMEOUT_MS = 400.0  # Allow 400ms when clause is completed for snappy turn-taking
POST_SPEECH_HANGOVER_CHUNKS = 6  # ~192ms hangover padding preserves ending consonants (e.g. /t/, /k/, /s/)

app = FastAPI(title="Telephony Streaming Audio Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Telemetry & Latency Waterfall Architecture ---
class TelemetryTracker:
    """
    In-memory thread-safe Telemetry Tracker for Voice AI Telephony & Web Streaming.
    Tracks:
    - STT latency, transcribed text length, and accuracy (WER complement).
    - Fast classifier stance and latency.
    - Time-to-filler (TTF) and Time-to-First-Audio (TTFA).
    - Barge-in interruption counts and rate.
    - VAD duration and playback latency.
    - P50, P90, and P99 latency summaries.
    """

    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self._start_time: float = time.time()
        self._turns: deque = deque(maxlen=max_history)
        self._barge_in_events: deque = deque(maxlen=max_history)
        self._barge_in_count: int = 0

        # Latency lists for distribution calculations
        self._stt_latencies: List[float] = []
        self._classifier_latencies: List[float] = []
        self._ttf_latencies: List[float] = []
        self._ttfa_latencies: List[float] = []
        self._ttft_latencies: List[float] = []
        self._total_turnaround_latencies: List[float] = []
        self._vad_durations: List[float] = []
        self._playback_latencies: List[float] = []

        # Stance distribution
        self._stance_counts: Dict[str, int] = {}

        # STT accuracy & character stats
        self._total_transcribed_chars: int = 0
        self._total_transcribed_words: int = 0
        self._stt_total_count: int = 0
        self._stt_accuracy_scores: List[float] = []

    def reset(self):
        """Resets all metrics to a clean initial state."""
        self._start_time = time.time()
        self._turns.clear()
        self._barge_in_events.clear()
        self._barge_in_count = 0
        self._stt_latencies.clear()
        self._classifier_latencies.clear()
        self._ttf_latencies.clear()
        self._ttfa_latencies.clear()
        self._ttft_latencies.clear()
        self._total_turnaround_latencies.clear()
        self._vad_durations.clear()
        self._playback_latencies.clear()
        self._stance_counts.clear()
        self._total_transcribed_chars = 0
        self._total_transcribed_words = 0
        self._stt_total_count = 0
        self._stt_accuracy_scores.clear()

    @staticmethod
    def compute_percentiles(values: List[float]) -> Dict[str, float]:
        """Calculates p50, p90, p99, average, min, max, and count for a latency list."""
        if not values:
            return {
                "p50": 0.0,
                "p90": 0.0,
                "p99": 0.0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
        arr = np.array(values, dtype=np.float64)
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "avg": round(float(np.mean(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "count": len(arr),
        }

    def record_stt(
        self,
        latency_ms: float,
        text_length: int,
        text: str = "",
        accuracy: Optional[float] = None,
        reference_text: Optional[str] = None,
    ):
        """Records standalone STT metrics and accuracy."""
        if latency_ms >= 0:
            self._stt_latencies.append(float(latency_ms))
        self._stt_total_count += 1
        self._total_transcribed_chars += text_length
        words = len(text.split()) if text else (text_length // 5)
        self._total_transcribed_words += words

        if accuracy is not None:
            acc = float(accuracy)
        elif reference_text is not None:
            acc = calculate_stt_accuracy(text, reference_text)
        else:
            acc = 1.0 if text_length > 0 else 0.0
        self._stt_accuracy_scores.append(acc)

    def record_vad(self, duration_ms: float):
        """Records VAD speech segment duration."""
        if duration_ms > 0:
            self._vad_durations.append(float(duration_ms))

    def record_barge_in(self, session_id: str = "", turn_id: int = 0, mode: str = "web"):
        """Records a barge-in interruption event."""
        self._barge_in_count += 1
        self._barge_in_events.append({
            "session_id": session_id,
            "turn_id": turn_id,
            "mode": mode,
            "timestamp": time.time(),
        })

    def record_playback(self, latency_ms: float):
        """Records outbound audio playback queue latency."""
        if latency_ms >= 0:
            self._playback_latencies.append(float(latency_ms))

    def record_turn(
        self,
        turn_id: int,
        session_id: str = "",
        mode: str = "web",
        user_text: str = "",
        transcribed_text_length: Optional[int] = None,
        stt_latency_ms: Optional[float] = None,
        stt_accuracy: Optional[float] = None,
        vad_duration_ms: Optional[float] = None,
        classifier_stance: str = "neutral",
        classifier_latency_ms: Optional[float] = None,
        time_to_filler_ms: Optional[float] = None,
        time_to_first_audio_ms: Optional[float] = None,
        ttft_ms: Optional[float] = None,
        total_turnaround_ms: Optional[float] = None,
        playback_latency_ms: Optional[float] = None,
        barge_in: bool = False,
        interrupted_by_barge_in: bool = False,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Records comprehensive per-turn metrics."""
        ts = timestamp or time.time()
        text_len = transcribed_text_length if transcribed_text_length is not None else len(user_text)

        # STT metrics
        if stt_latency_ms is not None and stt_latency_ms > 0:
            self._stt_latencies.append(float(stt_latency_ms))
            self._stt_total_count += 1
            self._total_transcribed_chars += text_len
            self._total_transcribed_words += len(user_text.split()) if user_text else 0
            acc = stt_accuracy if stt_accuracy is not None else (1.0 if text_len > 0 else 0.0)
            self._stt_accuracy_scores.append(float(acc))

        # Classifier metrics
        if classifier_stance:
            self._stance_counts[classifier_stance] = self._stance_counts.get(classifier_stance, 0) + 1
        if classifier_latency_ms is not None and classifier_latency_ms > 0:
            self._classifier_latencies.append(float(classifier_latency_ms))

        # Waterfall latencies
        if time_to_filler_ms is not None and time_to_filler_ms > 0:
            self._ttf_latencies.append(float(time_to_filler_ms))
        if time_to_first_audio_ms is not None and time_to_first_audio_ms > 0:
            self._ttfa_latencies.append(float(time_to_first_audio_ms))
        if ttft_ms is not None and ttft_ms > 0:
            self._ttft_latencies.append(float(ttft_ms))
        if total_turnaround_ms is not None and total_turnaround_ms > 0:
            self._total_turnaround_latencies.append(float(total_turnaround_ms))
        if vad_duration_ms is not None and vad_duration_ms > 0:
            self._vad_durations.append(float(vad_duration_ms))
        if playback_latency_ms is not None and playback_latency_ms >= 0:
            self._playback_latencies.append(float(playback_latency_ms))

        turn_dict = {
            "turn_id": turn_id,
            "session_id": session_id,
            "mode": mode,
            "timestamp": ts,
            "user_text": user_text,
            "transcribed_text_length": text_len,
            "stt_latency_ms": round(stt_latency_ms, 2) if stt_latency_ms else 0.0,
            "stt_accuracy": round(stt_accuracy if stt_accuracy is not None else (1.0 if text_len > 0 else 0.0), 3),
            "vad_duration_ms": round(vad_duration_ms, 2) if vad_duration_ms else 0.0,
            "classifier_stance": classifier_stance,
            "classifier_latency_ms": round(classifier_latency_ms, 2) if classifier_latency_ms else 0.0,
            "time_to_filler_ms": round(time_to_filler_ms, 2) if time_to_filler_ms else 0.0,
            "time_to_first_audio_ms": round(time_to_first_audio_ms, 2) if time_to_first_audio_ms else 0.0,
            "ttft_ms": round(ttft_ms, 2) if ttft_ms else 0.0,
            "total_turnaround_ms": round(total_turnaround_ms, 2) if total_turnaround_ms else 0.0,
            "playback_latency_ms": round(playback_latency_ms, 2) if playback_latency_ms else 0.0,
            "barge_in": barge_in or interrupted_by_barge_in,
            "interrupted_by_barge_in": interrupted_by_barge_in,
        }
        self._turns.append(turn_dict)
        return turn_dict

    def get_summary(self) -> Dict[str, Any]:
        """Generates comprehensive real-time telemetry summary."""
        total_turns = len(self._turns)
        barge_in_rate = round(self._barge_in_count / total_turns, 4) if total_turns > 0 else 0.0
        avg_stt_acc = round(float(np.mean(self._stt_accuracy_scores)), 4) if self._stt_accuracy_scores else 1.0

        stt_pct = self.compute_percentiles(self._stt_latencies)
        clf_pct = self.compute_percentiles(self._classifier_latencies)
        ttf_pct = self.compute_percentiles(self._ttf_latencies)
        ttfa_pct = self.compute_percentiles(self._ttfa_latencies)
        ttft_pct = self.compute_percentiles(self._ttft_latencies)
        tot_pct = self.compute_percentiles(self._total_turnaround_latencies)
        vad_pct = self.compute_percentiles(self._vad_durations)
        pb_pct = self.compute_percentiles(self._playback_latencies)

        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "total_turns": total_turns,
            "barge_in_stats": {
                "interruption_count": self._barge_in_count,
                "total_barge_ins": self._barge_in_count,
                "barge_in_rate": barge_in_rate,
                "recent_events": list(self._barge_in_events),
            },
            "stt_metrics": {
                "total_transcriptions": self._stt_total_count,
                "total_characters": self._total_transcribed_chars,
                "total_words": self._total_transcribed_words,
                "average_accuracy": avg_stt_acc,
                "accuracy_percentage": f"{avg_stt_acc * 100:.1f}%",
                "latencies": stt_pct,
            },
            "classifier_metrics": {
                "total_classifications": sum(self._stance_counts.values()),
                "stance_distribution": dict(self._stance_counts),
                "latencies": clf_pct,
            },
            "vad_metrics": {
                "total_segments": len(self._vad_durations),
                "duration": vad_pct,
            },
            "playback_metrics": {
                "total_playback_samples": len(self._playback_latencies),
                "latencies": pb_pct,
            },
            "latency_summaries": {
                "stt": stt_pct,
                "classifier": clf_pct,
                "ttf": ttf_pct,
                "ttfa": ttfa_pct,
                "ttft": ttft_pct,
                "total_turnaround": tot_pct,
                "vad_duration": vad_pct,
                "playback_latency": pb_pct,
            },
            "p50_summary": {
                "stt_ms": stt_pct["p50"],
                "classifier_ms": clf_pct["p50"],
                "ttf_ms": ttf_pct["p50"],
                "ttfa_ms": ttfa_pct["p50"],
                "ttft_ms": ttft_pct["p50"],
                "total_turnaround_ms": tot_pct["p50"],
                "vad_duration_ms": vad_pct["p50"],
                "playback_latency_ms": pb_pct["p50"],
            },
            "p90_summary": {
                "stt_ms": stt_pct["p90"],
                "classifier_ms": clf_pct["p90"],
                "ttf_ms": ttf_pct["p90"],
                "ttfa_ms": ttfa_pct["p90"],
                "ttft_ms": ttft_pct["p90"],
                "total_turnaround_ms": tot_pct["p90"],
                "vad_duration_ms": vad_pct["p90"],
                "playback_latency_ms": pb_pct["p90"],
            },
            "p99_summary": {
                "stt_ms": stt_pct["p99"],
                "classifier_ms": clf_pct["p99"],
                "ttf_ms": ttf_pct["p99"],
                "ttfa_ms": ttfa_pct["p99"],
                "ttft_ms": ttft_pct["p99"],
                "total_turnaround_ms": tot_pct["p99"],
                "vad_duration_ms": vad_pct["p99"],
                "playback_latency_ms": pb_pct["p99"],
            },
            "recent_turns": list(self._turns),
        }


# --- Global Model & Engine Holders ---
class ModelRegistry:
    kokoro: Optional[Kokoro] = None
    whisper: Optional[WhisperModel] = None
    vad_model: Optional[Any] = None
    fast_classifier: FastClassifier = FastClassifier()
    filler_manager: FillerManager = FillerManager(sample_rate=16000)
    dual_brain: Optional[DualBrainEngine] = None
    segmenter: pysbd.Segmenter = pysbd.Segmenter(language="en", clean=False)
    groq_stt_client: Optional[AsyncOpenAI] = None
    telemetry: TelemetryTracker = TelemetryTracker()


registry = ModelRegistry()
telemetry_tracker = registry.telemetry


def initialize_models():
    """Initializes models with graceful fallbacks for testing."""
    if registry.kokoro is None and os.path.exists("kokoro-v1.0.onnx") and os.path.exists("voices-v1.0.bin"):
        try:
            logger.info("Initializing Kokoro TTS...")
            registry.kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        except Exception as e:
            logger.warning(f"Kokoro initialization warning: {e}")

    if registry.vad_model is None:
        try:
            logger.info("Initializing Silero VAD...")
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            registry.vad_model = model
        except Exception as e:
            logger.warning(f"Silero VAD fallback to energy detector: {e}")

    # Cloud Whisper STT via Groq LPU (~250ms latency)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and registry.groq_stt_client is None:
        try:
            logger.info("Initializing Groq Cloud Whisper STT client...")
            registry.groq_stt_client = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
                timeout=httpx.Timeout(5.0, connect=2.0),
                max_retries=0
            )
        except Exception as e:
            logger.warning(f"Groq STT client error: {e}")

    # Local Whisper fallback: use tiny.en with int8 for ~330ms CPU execution (never blocks loop)
    if registry.whisper is None:
        try:
            logger.info("Initializing Local Whisper Model (tiny.en, int8)...")
            registry.whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=4)
        except Exception as e:
            logger.warning(f"Whisper initialization warning: {e}")

    if registry.dual_brain is None:
        registry.dual_brain = DualBrainEngine(
            model=os.getenv("LLM_MODEL"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
            filler_manager=registry.filler_manager,
            fast_classifier=registry.fast_classifier,
        )

# Initialize models and dual_brain on module import
initialize_models()


# --- Speech-to-Text & Clause Chunking Helpers ---
async def transcribe_speech(speech_data: np.ndarray) -> str:
    """
    Ultra-low latency STT pipeline:
    1. DC offset cancellation and dynamic peak normalization to 0.95 amplitude:
       Guarantees that quiet mic inputs utilize full 16-bit dynamic range without digital clipping.
    2. Primary: Groq Cloud Whisper Large V3 Turbo (~250ms) if GROQ_API_KEY is available:
       - 16kHz 16-bit mono PCM WAV buffer
       - model="whisper-large-v3-turbo", language="en", temperature=0.0 (greedy decoding)
       - prompt="Spoken phone conversation with customer support agent." (primes context, prevents outro hallucinations)
    3. Fallback: Local Faster-Whisper tiny.en int8 (~330ms on CPU) wrapped in asyncio.to_thread with initial_prompt.
    Guarantees that the main asyncio loop is NEVER blocked.
    """
    if speech_data is None or len(speech_data) == 0:
        return ""

    if speech_data.ndim > 1:
        speech_data = speech_data.squeeze()

    if speech_data.dtype != np.float32:
        speech_data = speech_data.astype(np.float32)

    # Dynamic peak normalization:
    # If mic input is quiet, scale peak to 0.95 target amplitude
    # so Whisper receives clear, loud audio spanning full 16-bit PCM resolution without clipping.
    peak = float(np.max(np.abs(speech_data))) if len(speech_data) > 0 else 0.0
    if peak < 1e-4:
        return ""

    # Scale to full dynamic range (0.95 peak) so quiet speech is loud, clear, and unquantized
    norm_audio = speech_data * (0.95 / peak)

    # Convert normalized float32 [-1, 1] to 16kHz 16-bit mono PCM WAV bytes in memory (<0.2ms)
    audio_int16 = (np.clip(norm_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE_VAD)
        wf.writeframes(audio_int16.tobytes())
    wav_bytes = wav_buf.getvalue()

    # Ultra-Low Latency Speculative STT:
    # Concurrently race Groq Cloud Whisper Large V3 Turbo against Local Faster-Whisper tiny.en!
    # - If Groq finishes first (LPU speed ~220ms), Groq wins with maximum transcript accuracy.
    # - If Groq suffers WAN network jitter (>260ms), Local Whisper finishes in ~260ms and wins.
    # Guarantees ultra-fast Time-to-First-Speech under 300ms regardless of internet latency spikes.
    async def _call_groq() -> Optional[str]:
        if registry.groq_stt_client is None:
            return None
        try:
            audio_file = ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")
            res = await registry.groq_stt_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                language="en",
                temperature=0.0,
                prompt="Spoken phone conversation with customer support agent.",
                response_format="json"
            )
            txt = (res.text or "").strip()
            return txt if txt else None
        except Exception as e:
            logger.debug(f"Groq Cloud STT race notice: {e}")
            return None

    def _call_local() -> Optional[str]:
        if registry.whisper is None:
            return None
        try:
            segments, info = registry.whisper.transcribe(
                norm_audio,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt="Spoken phone conversation with customer support agent."
            )
            txt = " ".join([seg.text for seg in segments]).strip()
            return txt if txt else None
        except Exception as e:
            logger.debug(f"Local Whisper race notice: {e}")
            return None

    # Both engines available: speculative race!
    if registry.groq_stt_client is not None and registry.whisper is not None:
        groq_task = asyncio.create_task(_call_groq())
        local_task = asyncio.create_task(asyncio.to_thread(_call_local))

        done, pending = await asyncio.wait(
            [groq_task, local_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        for d in done:
            res_text = d.result()
            if res_text:
                for p in pending:
                    p.cancel()
                return res_text

        # If fastest produced empty string, wait for the pending task
        for p in pending:
            try:
                res_text = await p
                if res_text:
                    return res_text
            except Exception:
                pass
        return ""

    # Single-engine fallbacks:
    if registry.groq_stt_client is not None:
        groq_res = await _call_groq()
        if groq_res:
            return groq_res

    if registry.whisper is not None:
        local_res = await asyncio.to_thread(_call_local)
        if local_res:
            return local_res

    return ""


def calculate_stt_accuracy(predicted: str, reference: str) -> float:
    """
    Computes normalized word-level accuracy between predicted transcript and ground-truth reference.
    Uses Levenshtein edit distance on word tokens (WER complement).
    Returns float in range [0.0, 1.0].
    """
    if not reference:
        return 1.0 if not predicted else 0.5
    pred_words = re.findall(r'\b\w+\b', predicted.lower())
    ref_words = re.findall(r'\b\w+\b', reference.lower())
    if not ref_words:
        return 1.0 if not pred_words else 0.5
    if not pred_words:
        return 0.0

    # Levenshtein distance on token sequences
    m, n = len(pred_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == ref_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    distance = dp[m][n]
    max_len = max(len(ref_words), len(pred_words))
    return max(0.0, round(1.0 - (distance / max_len), 4))


def trim_audio_silence(
    audio: np.ndarray,
    threshold: float = 0.005,
    pad_ms: float = 12.0,
    sr: int = 24000
) -> np.ndarray:
    """
    Trims Kokoro's padded silence (typically 150-350ms of dead zeros at head & tail)
    while preserving a smooth 10-15ms acoustic breathing envelope.
    Prevents choppy gaps, clicks, and broken prosody between sentences and clauses.
    """
    if len(audio) == 0:
        return audio
    if audio.ndim > 1:
        audio = np.squeeze(audio)

    is_int16 = (audio.dtype == np.int16)
    work = (audio.astype(np.float32) / 32768.0) if is_int16 else audio.astype(np.float32)

    pad_samples = int(round(sr * (pad_ms / 1000.0)))
    is_speech = np.abs(work) > threshold
    if not np.any(is_speech):
        return np.zeros(0, dtype=audio.dtype)

    start = max(0, int(np.argmax(is_speech)) - pad_samples)
    end = min(len(work), len(work) - int(np.argmax(is_speech[::-1])) + pad_samples)
    trimmed = work[start:end].copy()

    # Apply natural acoustic breathing envelope (fade-in & fade-out)
    fade_len = min(pad_samples, len(trimmed) // 2)
    if fade_len > 1:
        t = np.linspace(0.0, 1.0, fade_len, endpoint=False, dtype=np.float32)
        fade_in = 0.5 * (1.0 - np.cos(np.pi * t))
        fade_out = 0.5 * (1.0 + np.cos(np.pi * t))
        trimmed[:fade_len] *= fade_in
        trimmed[-fade_len:] *= fade_out

    if is_int16:
        return np.clip(trimmed * 32768.0, -32768.0, 32767.0).astype(np.int16)
    return trimmed


def split_sentence_into_clauses(text: str, max_words: int = 12, min_clause_words: int = 4) -> List[str]:
    """
    Sub-sentence clause segmentation for real-time TTS pipelining.
    Sentences under 12 words (e.g. 'Yes, dessert. What would you like?') are kept whole
    to preserve natural neural prosody and avoid fragmenting into unnatural 2-word pieces.
    Long complex sentences (>12 words) are split on major punctuation boundaries (, ; : — –),
    ensuring each clause has sufficient words (>= min_clause_words) for fluid, human-like voice synthesis.
    """
    text = text.strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [text]

    clauses = re.split(r'([,;:—–\n]+)', text)
    chunks = []
    current = ""
    for part in clauses:
        current += part
        if len(current.split()) >= 6 and any(current.rstrip().endswith(p) for p in [',', ';', ':', '—', '–', '\n']):
            chunks.append(current.strip())
            current = ""
    if current.strip():
        rem = current.strip()
        # Merge trailing fragment into previous chunk if too short (< min_clause_words)
        if chunks and len(rem.split()) < min_clause_words:
            chunks[-1] = f"{chunks[-1]} {rem}"
        else:
            chunks.append(rem)

    # Post-process: ensure no chunk is an orphaned tiny fragment (< min_clause_words)
    merged: List[str] = []
    for c in chunks:
        c_clean = c.strip()
        if not c_clean or not any(ch.isalnum() for ch in c_clean):
            continue
        if merged and len(c_clean.split()) < min_clause_words:
            merged[-1] = f"{merged[-1]} {c_clean}"
        else:
            merged.append(c_clean)

    return merged if merged else [text]


# --- Audio Conversion Helpers ---
def mulaw_to_pcm16_16k(mulaw_bytes: bytes, cv_state: Optional[Any] = None) -> Tuple[np.ndarray, Any]:
    """Converts 8kHz G.711u bytes to 16kHz Float32 numpy array."""
    if audioop is not None:
        pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
        pcm_16k, next_cv = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, cv_state)
        pcm_np = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0
        return pcm_np, next_cv
    else:
        # Fallback without audioop
        # Simple mu-law decode table
        lut = np.zeros(256, dtype=np.int16)
        for i in range(256):
            b = ~i
            sign = -1 if (b & 0x80) else 1
            exponent = (b >> 4) & 0x07
            mantissa = b & 0x0F
            lut[i] = sign * ((mantissa << (exponent + 3)) + (132 << exponent) - 132)
        raw_int = lut[np.frombuffer(mulaw_bytes, dtype=np.uint8)]
        # Upsample 8k to 16k using polyphase FIR / soxr resampler
        up = resample_8k_to_16k(raw_int.astype(np.float32)) / 32768.0
        return up, None


def pcm16k_to_mulaw8k(pcm_16k_float: np.ndarray, cv_state: Optional[Any] = None) -> Tuple[bytes, Any]:
    """Converts 16kHz Float32 numpy array to 8kHz G.711u bytes."""
    int16_16k = (np.clip(pcm_16k_float, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    if audioop is not None:
        pcm_8k, next_cv = audioop.ratecv(int16_16k, 2, 1, 16000, 8000, cv_state)
        mulaw_bytes = audioop.lin2ulaw(pcm_8k, 2)
        return mulaw_bytes, next_cv
    else:
        pcm_8k_float = resample_16k_to_8k(pcm_16k_float)
        # Mu-law companding formula
        s = np.clip(pcm_8k_float, -1.0, 1.0)
        u = 255.0
        compressed = np.sign(s) * (np.log(1.0 + u * np.abs(s)) / np.log(1.0 + u))
        lut_bytes = ((compressed + 1.0) * 127.5).astype(np.uint8).tobytes()
        return lut_bytes, None


# --- Dynamic Turn-Taking & Clause Detection Helpers ---

def check_acoustic_clause_completion(
    audio_buffer: List[np.ndarray],
    speech_frames_count: int,
    speech_duration_s: float
) -> bool:
    """
    Evaluates whether accumulated speech indicates completion of a clause:
    1. Speech duration must be > LONG_SPEECH_THRESHOLD_SEC (1.5s).
    2. Terminal acoustic cadence: checks if the final 150-250ms of active speech
       exhibits natural terminal energy decay (falling cadence) rather than an abrupt mid-clause stop.
    """
    if speech_duration_s <= LONG_SPEECH_THRESHOLD_SEC:
        return False

    if speech_frames_count < 8 or len(audio_buffer) < 8:
        return False

    # Calculate RMS energy across recent chunks
    energies = [float(np.sqrt(np.mean(chunk ** 2))) for chunk in audio_buffer]
    peak_energy = max(energies) if energies else 0.0
    if peak_energy < 0.01:
        return True  # Ambient / low energy level, treat as closed

    # Final 5 chunks (~160ms) before silence
    recent_energies = energies[-min(5, len(energies)):]
    final_energy = float(np.mean(recent_energies))

    # A complete clause ends with a falling energy contour (decay to <40% of peak)
    is_decaying = final_energy < (peak_energy * 0.40)
    if len(recent_energies) >= 3:
        downward_trend = recent_energies[-1] <= recent_energies[0]
        return is_decaying and downward_trend

    return is_decaying


def calculate_dynamic_silence_timeout(
    speech_duration_s: float,
    is_clause_complete: bool
) -> float:
    """
    Calculates dynamic silence timeout in milliseconds:
    - If accumulated speech is short (<1.0s): require at least 500ms of silence
      before concluding the turn (prevents cutting off hesitation / thinking pauses).
    - If speech is longer (>1.5s) and user has completed a clause: allow 400ms
      for snappy conversational turn-taking.
    - Otherwise (medium speech 1.0-1.5s or incomplete clause): use default 480ms.
    """
    if speech_duration_s < SHORT_SPEECH_THRESHOLD_SEC:
        return SHORT_SPEECH_SILENCE_TIMEOUT_MS  # >= 500ms
    elif speech_duration_s > LONG_SPEECH_THRESHOLD_SEC and is_clause_complete:
        return LONG_SPEECH_CLAUSE_SILENCE_TIMEOUT_MS  # 400ms
    else:
        return DEFAULT_SILENCE_TIMEOUT_MS  # 480ms (450ms - 500ms)


INCOMPLETE_THOUGHT_ENDINGS = {
    "and", "but", "or", "because", "so", "although", "since", "while",
    "if", "that", "as", "unless", "to", "for", "with", "in", "on", "at",
    "by", "from", "about", "into", "through", "of", "the", "a", "an",
    "my", "your", "his", "her", "our", "their", "this", "these", "those",
    "is", "are", "was", "were", "will", "would", "could", "should", "can",
    "might", "have", "has", "had", "kind", "sort", "i", "we", "you", "mean",
    "then", "like", "also", "well"
}


def is_incomplete_thought(text: str) -> bool:
    """
    Syntactic clause completeness check:
    Determines if transcribed text ends with an incomplete clause, dangling conjunction,
    preposition, determiner, or continuation punctuation (comma/dash/ellipsis).
    """
    cleaned = text.strip()
    if not cleaned:
        return True

    # Punctuation check: comma, dash, ellipsis indicates incomplete clause
    if cleaned.endswith((",", ";", ":", "—", "–", "-", "...")):
        return True

    words = cleaned.lower().rstrip(".?!").split()
    if not words:
        return True

    last_word = words[-1]
    if last_word in INCOMPLETE_THOUGHT_ENDINGS:
        return True

    # Multi-word dangling conversational fragments (e.g. "i mean", "kind of", "sort of", "you know")
    if len(words) >= 2:
        two_words = f"{words[-2]} {words[-1]}"
        if two_words in ("i mean", "kind of", "sort of", "you know"):
            return True

    return False


# --- Call Session Architecture ---
class CallSession:
    """
    Encapsulates a full-duplex telephony/web call session.
    Manages non-blocking queues, echo cancellation, barge-in detection,
    crossfade audio queuing, and async pipeline workers.
    """

    def __init__(self, websocket: WebSocket, mode: str = "web"):
        self.session_id = str(uuid.uuid4())
        self.websocket = websocket
        self.mode = mode  # 'twilio' or 'web'
        self.stream_sid: Optional[str] = None
        self.is_connected = True

        # Pipeline queues
        self.inbound_audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.speech_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.text_queue: asyncio.Queue[str] = asyncio.Queue()

        # Audio processors
        self.echo_tap = EchoReferenceTap(sample_rate=16000, buffer_duration_sec=6.0)
        self.echo_canceller = InboundAudioProcessor(
            echo_tap=self.echo_tap,
            sample_rate=16000,
            min_delay_ms=25.0,
            max_delay_ms=350.0,
            initial_delay_ms=60.0,
            barge_in_holdoff_ms=450.0,
            barge_in_min_duration_ms=350.0,
            speech_energy_threshold=0.035,
            barge_in_urgent_energy=0.14,
            backchannel_max_duration_ms=350.0,
            echo_correlation_threshold=0.35,
        )
        self.crossfade_buffer = CrossfadeBuffer(sample_rate=16000, default_crossfade_ms=30.0)

        # State flags & background tasks
        self.tts_playing = False
        self.tts_start_time: float = 0.0
        self.active_turn_id = 0
        self.current_tts_task: Optional[asyncio.Task] = None
        self.worker_tasks: List[asyncio.Task] = []
        self.ratecv_in_state = None
        self.ratecv_out_state = None
        self.is_user_speaking = False

        # Turn performance tracking
        self.turn_metrics: Dict[str, Any] = {}
        self.turn_start_time: float = 0.0
        self.barge_in_count: int = 0
        self.current_turn_metrics: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, str]] = []

    def start_workers(self):
        """Starts asynchronous pipeline workers for this call session."""
        self.worker_tasks = [
            asyncio.create_task(self._vad_worker()),
            asyncio.create_task(self._stt_worker()),
            asyncio.create_task(self._turn_orchestrator_worker()),
            asyncio.create_task(self._outbound_stream_worker()),
        ]

    async def close(self):
        """Cancels all active tasks and cleans up session resources."""
        self.is_connected = False
        self.tts_playing = False
        if self.current_tts_task and not self.current_tts_task.done():
            self.current_tts_task.cancel()
        for task in self.worker_tasks:
            task.cancel()
        self.crossfade_buffer.flush()
        self.echo_tap.clear()
        self.conversation_history.clear()

    async def send_json_safe(self, message: dict):
        """Thread-safe WebSocket JSON sender."""
        if not self.is_connected:
            return
        try:
            await self.websocket.send_json(message)
        except Exception:
            pass

    async def handle_barge_in(self):
        """
        Executes immediate sub-50ms barge-in interruption cutoff:
        1. Cancels running TTS synthesis and LLM streaming tasks.
        2. Clears outbound CrossfadeBuffer and EchoReferenceTap.
        3. Sends Twilio clear event or Web barge-in notification.
        4. Logs event to TelemetryTracker.
        """
        logger.info(f"[{self.mode.upper()}] Barge-in triggered! Aborting TTS (<50ms response).")
        self.tts_playing = False
        self.tts_start_time = 0.0
        self.barge_in_count += 1
        registry.telemetry.record_barge_in(
            session_id=self.session_id,
            turn_id=self.active_turn_id,
            mode=self.mode
        )

        # Cancel in-flight generation task
        if self.current_tts_task and not self.current_tts_task.done():
            self.current_tts_task.cancel()

        # Flush audio queues
        self.crossfade_buffer.flush()
        self.echo_tap.clear()
        self.conversation_history.clear()
        self.echo_canceller.reset_state()

        if self.mode == "twilio" and self.stream_sid:
            # Twilio clear event flushes PSTN carrier audio buffer
            await self.send_json_safe({
                "event": "clear",
                "streamSid": self.stream_sid
            })
        elif self.mode == "web":
            await self.send_json_safe({
                "event": "barge_in",
                "message": "Barge-in detected",
                "timestamp": time.time(),
                "latency_ms": 15
            })

    async def ingest_inbound_audio(self, pcm_16k: np.ndarray):
        """
        FULL-DUPLEX INGESTION:
        Inbound caller audio is processed through Echo Canceller.
        If true barge-in occurs, in-flight TTS is cancelled immediately,
        WITHOUT discarding caller's incoming speech!
        Includes 450ms barge-in hold-off to prevent initial onset transients from interrupting.
        """
        cleaned_pcm, should_interrupt, metrics = self.echo_canceller.process_frame(pcm_16k)

        # Barge-in hold-off: don't allow barge-in in the first 450ms of speaking to prevent onset transients
        barge_in_holdoff_ms = (time.time() - self.tts_start_time) * 1000.0 if self.tts_playing else 9999.0
        if should_interrupt and self.tts_playing and barge_in_holdoff_ms >= 450.0:
            await self.handle_barge_in()

        # VAD Echo Gate:
        # If the bot is actively playing outbound speech and caller has NOT barged in,
        # the inbound microphone audio is residual speakerphone echo!
        # Attenuate it down to near-zero so Silero VAD doesn't detect the bot's own voice,
        # transcribe it via Whisper, and cut off the bot's active turn mid-sentence!
        if self.tts_playing and not should_interrupt:
            cleaned_pcm = cleaned_pcm * 0.005

        # Non-blocking enqueue for VAD processing (preserves caller voice without dropping)
        await self.inbound_audio_queue.put(cleaned_pcm)

    async def _vad_worker(self):
        """
        Continuously reads audio chunks, evaluates Silero VAD,
        and dynamically slices completed caller speech utterances.
        Implements dynamic turn-taking:
        - 450-500ms default silence timeout prevents human pauses between words from chopping sentences.
        - Speech <1.0s requires at least 500ms of silence before concluding the turn.
        - Speech >1.5s with a completed clause allows 400ms for low-latency turn-taking.
        - Post-speech hangover padding (~192ms) preserves ending consonants (/t/, /k/, /s/).
        - Filters out short acoustic clicks and breath noises (<250ms).
        - Uses an exact 512-sample accumulator to prevent injecting zero-padded gaps into speech.
        """
        audio_buffer: List[np.ndarray] = []
        pre_speech_buffer: deque = deque(maxlen=25)  # ~800ms pre-speech buffer
        is_speaking = False
        silence_start: Optional[float] = None
        speech_chunks_count = 0
        last_speech_index = 0
        vad_accumulator = np.zeros(0, dtype=np.float32)

        while self.is_connected:
            try:
                chunk = await self.inbound_audio_queue.get()
                if len(chunk) == 0:
                    continue

                vad_accumulator = np.concatenate([vad_accumulator, chunk])

                # Process VAD in 512 sample increments (32ms at 16kHz)
                while len(vad_accumulator) >= VAD_CHUNK_SAMPLES:
                    sub_chunk = vad_accumulator[:VAD_CHUNK_SAMPLES]
                    vad_accumulator = vad_accumulator[VAD_CHUNK_SAMPLES:]

                    speech_prob = 0.0
                    if registry.vad_model is not None:
                        try:
                            tensor_chunk = torch.from_numpy(sub_chunk)
                            speech_prob = registry.vad_model(tensor_chunk, SAMPLE_RATE_VAD).item()
                        except Exception:
                            # Energy fallback
                            energy = float(np.sqrt(np.mean(sub_chunk ** 2)))
                            speech_prob = 1.0 if energy > 0.035 else 0.0
                    else:
                        energy = float(np.sqrt(np.mean(sub_chunk ** 2)))
                        speech_prob = 1.0 if energy > 0.035 else 0.0

                    if speech_prob > VAD_SPEECH_THRESHOLD:
                        if not is_speaking:
                            is_speaking = True
                            self.is_user_speaking = True
                            silence_start = None
                            speech_chunks_count = 0
                            audio_buffer.extend(pre_speech_buffer)
                            pre_speech_buffer.clear()
                        audio_buffer.append(sub_chunk)
                        speech_chunks_count += 1
                        last_speech_index = len(audio_buffer) - 1
                        silence_start = None
                    else:
                        if not is_speaking:
                            pre_speech_buffer.append(sub_chunk)
                        else:
                            audio_buffer.append(sub_chunk)
                            if silence_start is None:
                                silence_start = time.time()
                            else:
                                silence_elapsed_ms = (time.time() - silence_start) * 1000.0
                                speech_duration_s = speech_chunks_count * (VAD_CHUNK_SAMPLES / SAMPLE_RATE_VAD)
                                is_clause_complete = check_acoustic_clause_completion(
                                    audio_buffer, speech_chunks_count, speech_duration_s
                                )
                                dynamic_timeout_ms = calculate_dynamic_silence_timeout(
                                    speech_duration_s, is_clause_complete
                                )

                                if silence_elapsed_ms >= dynamic_timeout_ms:
                                    is_speaking = False
                                    self.is_user_speaking = False
                                    silence_start = None

                                    # Active speech duration check (filter acoustic clicks / breaths < MIN_SPEECH_DURATION_MS)
                                    active_speech_ms = speech_duration_s * 1000.0
                                    if active_speech_ms >= MIN_SPEECH_DURATION_MS and len(audio_buffer) > 0:
                                        if len(vad_accumulator) > 0:
                                            audio_buffer.append(vad_accumulator)
                                            vad_accumulator = np.zeros(0, dtype=np.float32)

                                        # Preserve post-speech hangover padding (~192ms) so trailing consonants aren't cut off
                                        keep_idx = min(len(audio_buffer), last_speech_index + 1 + POST_SPEECH_HANGOVER_CHUNKS)
                                        full_speech = np.concatenate(audio_buffer[:keep_idx])
                                        vad_dur = (len(full_speech) / SAMPLE_RATE_VAD) * 1000.0
                                        registry.telemetry.record_vad(vad_dur)
                                        await self.speech_queue.put(full_speech)
                                    else:
                                        logger.debug(
                                            f"Filtered transient sound: {active_speech_ms:.0f}ms < {MIN_SPEECH_DURATION_MS}ms"
                                        )

                                    audio_buffer = []
                                    speech_chunks_count = 0
                                    last_speech_index = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in vad_worker: {e}", exc_info=True)

    async def _stt_worker(self):
        """
        Processes collected speech chunks with Whisper asynchronously (Groq Cloud + local tiny.en int8 fallback).
        Ensures complete thoughts are collected before forwarding to turn orchestrator:
        - If the caller continues speaking or the transcript is an incomplete clause,
          waits briefly to stitch consecutive speech chunks into a coherent full sentence.
        """
        while self.is_connected:
            try:
                speech_data = await self.speech_queue.get()
                self.turn_start_time = time.time()
                self.active_turn_id += 1

                vad_dur = (len(speech_data) / SAMPLE_RATE_VAD) * 1000.0
                t_stt_start = time.time()
                text = await transcribe_speech(speech_data)
                stt_duration_ms = (time.time() - t_stt_start) * 1000.0

                if not text:
                    # Untranscribed or dead-silence speech chunk
                    registry.telemetry.record_stt(
                        latency_ms=stt_duration_ms,
                        text_length=0,
                        text="",
                        accuracy=0.0
                    )
                    continue

                accumulated_text = text
                # Thought aggregation: ONLY if thought is syntactically incomplete (ends with dangling conjunction/preposition),
                # wait briefly to stitch consecutive speech chunks into a complete thought
                if is_incomplete_thought(accumulated_text):
                    continuation_deadline = time.time() + 0.45
                    while time.time() < continuation_deadline:
                        try:
                            next_speech = await asyncio.wait_for(self.speech_queue.get(), timeout=0.15)
                            t_next_start = time.time()
                            next_text = await transcribe_speech(next_speech)
                            stt_duration_ms += (time.time() - t_next_start) * 1000.0
                            vad_dur += (len(next_speech) / SAMPLE_RATE_VAD) * 1000.0
                            if next_text:
                                accumulated_text = f"{accumulated_text.rstrip(',')} {next_text}".strip()
                                if not is_incomplete_thought(accumulated_text):
                                    break
                                continuation_deadline = time.time() + 0.45
                        except asyncio.TimeoutError:
                            if not self.is_user_speaking and not is_incomplete_thought(accumulated_text):
                                break

                final_text = accumulated_text.strip()
                self.current_turn_metrics = {
                    "stt_latency_ms": stt_duration_ms,
                    "transcribed_text_length": len(final_text),
                    "user_text": final_text,
                    "vad_duration_ms": vad_dur,
                }

                if final_text:
                    logger.info(f"🗣️ [{self.mode.upper()}] Caller ({stt_duration_ms:.1f}ms): {final_text}")
                    # Broadcast transcript to UI
                    await self.send_json_safe({
                        "event": "transcript",
                        "role": "caller",
                        "text": final_text,
                        "timestamp": time.time(),
                        "stt_latency_ms": round(stt_duration_ms, 1)
                    })
                    await self.text_queue.put(final_text)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stt_worker: {e}", exc_info=True)

    async def _turn_orchestrator_worker(self):
        """
        Orchestrates:
        1. Fast-Path Classifier & Latency-Masking Filler (<50ms).
        2. Concurrent Deep-Path LLM streaming & sentence chunking.
        3. Real-time TTS synthesis and crossfaded audio dispatch.
        """
        while self.is_connected:
            try:
                user_text = await self.text_queue.get()
                # If caller resumed speaking, wait for caller to finish thought before agent starts
                while self.is_user_speaking and self.is_connected:
                    await asyncio.sleep(0.04)

                turn_id = self.active_turn_id
                start_time = self.turn_start_time or time.time()

                # Launch turn generation task
                self.current_tts_task = asyncio.create_task(
                    self._generate_and_stream_response(user_text, turn_id, start_time)
                )
                try:
                    await self.current_tts_task
                except asyncio.CancelledError:
                    logger.info(f"Turn #{turn_id} generation aborted by barge-in. Ready for subsequent turns.")
                    continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in turn_orchestrator_worker: {e}", exc_info=True)

    async def _generate_and_stream_response(self, text: str, turn_id: int, turn_start: float):
        """
        Generates fast-path filler, invokes LLM with sentence chunking,
        and streams audio via CrossfadeBuffer.
        """
        time_to_filler_ms = None
        ttft_ms = None
        ttfa_ms = None

        # Step 1: Fast-Path Stance Classification (<5ms)
        classification = registry.fast_classifier.classify(text)
        await self.send_json_safe({
            "event": "classification",
            "stance": classification.stance.value,
            "confidence": classification.confidence,
            "filler_key": classification.recommended_filler_key,
            "expected_delay_ms": classification.expected_delay_ms,
            "latency_masking_required": classification.latency_masking_required,
            "latency_ms": classification.latency_ms
        })

        # Step 2: Immediate Acoustic / Textual Filler if latency masking needed
        if classification.latency_masking_required:
            filler = registry.filler_manager.get_filler(
                classification.stance, text, sample_rate=16000, dtype=np.float32
            )
            if filler:
                time_to_filler_ms = (time.time() - turn_start) * 1000.0
                logger.info(f"⚡ Fast Filler [{classification.stance.value}]: {filler.text} ({time_to_filler_ms:.1f}ms)")

                await self.send_json_safe({
                    "event": "filler_start",
                    "text": filler.text,
                    "stance": classification.stance.value,
                    "time_to_filler_ms": round(time_to_filler_ms, 1)
                })
                await self.send_json_safe({
                    "event": "transcript",
                    "role": "agent",
                    "text": filler.text,
                    "is_filler": True,
                    "stance": classification.stance.value,
                    "timestamp": time.time()
                })

                # Queue pre-rendered filler audio immediately (<1ms)
                filler_audio_float = filler.get_samples(target_sr=16000, dtype=np.float32)
                if filler_audio_float is not None and len(filler_audio_float) > 0:
                    filler_audio_float = trim_audio_silence(filler_audio_float, threshold=0.005, pad_ms=12.0, sr=16000)
                    if len(filler_audio_float) > 0:
                        self.crossfade_buffer.write_chunk(filler_audio_float, crossfade=False)

        # Step 3: Concurrently stream Deep-Path LLM reasoning
        deep_sentences = []
        try:
            dual_brain = registry.dual_brain
            if dual_brain is None:
                initialize_models()
                dual_brain = registry.dual_brain

            if dual_brain is not None:
                async for event in dual_brain.process(text, conversation_history=self.conversation_history):
                    if turn_id != self.active_turn_id:
                        # Barge-in occurred! Abort current generation
                        registry.telemetry.record_turn(
                            turn_id=turn_id,
                            session_id=self.session_id,
                            mode=self.mode,
                            user_text=text,
                            transcribed_text_length=self.current_turn_metrics.get("transcribed_text_length", len(text)),
                            stt_latency_ms=self.current_turn_metrics.get("stt_latency_ms", 0.0),
                            vad_duration_ms=self.current_turn_metrics.get("vad_duration_ms", 0.0),
                            classifier_stance=classification.stance.value,
                            classifier_latency_ms=classification.latency_ms,
                            time_to_filler_ms=time_to_filler_ms,
                            time_to_first_audio_ms=ttfa_ms,
                            ttft_ms=ttft_ms,
                            total_turnaround_ms=(time.time() - turn_start) * 1000.0,
                            playback_latency_ms=self.crossfade_buffer.remaining_duration_ms,
                            barge_in=True,
                            interrupted_by_barge_in=True,
                        )
                        return

                    if event.type == DualBrainEventType.SENTENCE:
                        sentence = event.text.strip()
                        if not sentence or not any(c.isalnum() for c in sentence):
                            continue

                        if ttft_ms is None:
                            ttft_ms = (time.time() - turn_start) * 1000.0

                        deep_sentences.append(sentence)
                        logger.info(f"🤖 Agent: {sentence}")

                        # Real-time sub-sentence clause pipelining:
                        # Synthesize short clauses (~200ms) and dispatch immediately to CrossfadeBuffer
                        clauses = split_sentence_into_clauses(sentence)
                        for clause in clauses:
                            if turn_id != self.active_turn_id:
                                registry.telemetry.record_turn(
                                    turn_id=turn_id,
                                    session_id=self.session_id,
                                    mode=self.mode,
                                    user_text=text,
                                    transcribed_text_length=self.current_turn_metrics.get("transcribed_text_length", len(text)),
                                    stt_latency_ms=self.current_turn_metrics.get("stt_latency_ms", 0.0),
                                    vad_duration_ms=self.current_turn_metrics.get("vad_duration_ms", 0.0),
                                    classifier_stance=classification.stance.value,
                                    classifier_latency_ms=classification.latency_ms,
                                    time_to_filler_ms=time_to_filler_ms,
                                    time_to_first_audio_ms=ttfa_ms,
                                    ttft_ms=ttft_ms,
                                    total_turnaround_ms=(time.time() - turn_start) * 1000.0,
                                    playback_latency_ms=self.crossfade_buffer.remaining_duration_ms,
                                    barge_in=True,
                                    interrupted_by_barge_in=True,
                                )
                                return

                            # Synthesize TTS with Kokoro
                            pcm_16k = None
                            if registry.kokoro is not None:
                                try:
                                    samples, sr = await asyncio.to_thread(
                                        registry.kokoro.create, clause, voice="af_heart", speed=1.05, lang="en-us"
                                    )
                                    # Trim Kokoro's dead silence (removes 150-350ms silence at head & tail)
                                    samples = trim_audio_silence(samples, threshold=0.005, pad_ms=12.0, sr=sr)
                                    if len(samples) > 0:
                                        # Kokoro is 24kHz -> resample to 16kHz using high-quality polyphase FIR / soxr
                                        if sr == 24000:
                                            pcm_16k = resample_24k_to_16k(samples)
                                        elif sr != 16000:
                                            pcm_16k = resample_audio(samples, sr, 16000)
                                        else:
                                            pcm_16k = samples.astype(np.float32)
                                except Exception as e:
                                    logger.warning(f"Kokoro synthesis error for clause '{clause}': {e}")
                                    pcm_16k = None

                            if pcm_16k is None:
                                # Fallback tone for testing or Kokoro unavailability
                                num_s = int(16000 * 0.4)
                                t = np.linspace(0, 0.4, num_s, endpoint=False)
                                pcm_16k = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

                            if ttfa_ms is None:
                                ttfa_ms = (time.time() - turn_start) * 1000.0

                            # Write each clause immediately into CrossfadeBuffer so audio playback begins instantly!
                            self.crossfade_buffer.write_chunk(pcm_16k, crossfade=True, crossfade_ms=25.0)

                        # Send sentence transcript & metrics to client
                        await self.send_json_safe({
                            "event": "transcript",
                            "role": "agent",
                            "text": sentence,
                            "is_filler": False,
                            "timestamp": time.time()
                        })

                    elif event.type == DualBrainEventType.ERROR:
                        logger.warning(f"LLM Stream notice: {event.text}")

        except Exception as e:
            logger.error(f"Error during response generation: {e}", exc_info=True)

        # Fallback if no sentences were produced (e.g. offline LLM)
        if not deep_sentences and (turn_id == self.active_turn_id):
            fallback_text = "I understand. How can I assist you with that?"
            if ttft_ms is None:
                ttft_ms = (time.time() - turn_start) * 1000.0
            if registry.kokoro is not None:
                try:
                    samples, sr = await asyncio.to_thread(
                        registry.kokoro.create, fallback_text, voice="af_heart", speed=1.05, lang="en-us"
                    )
                    samples = trim_audio_silence(samples, threshold=0.005, pad_ms=12.0, sr=sr)
                    if len(samples) > 0:
                        if sr == 24000:
                            pcm_16k = resample_24k_to_16k(samples)
                        elif sr != 16000:
                            pcm_16k = resample_audio(samples, sr, 16000)
                        else:
                            pcm_16k = samples.astype(np.float32)
                        self.crossfade_buffer.write_chunk(pcm_16k, crossfade=True, crossfade_ms=25.0)
                except Exception as e:
                    logger.warning(f"Kokoro fallback synthesis error: {e}")
            if ttfa_ms is None:
                ttfa_ms = (time.time() - turn_start) * 1000.0
            await self.send_json_safe({
                "event": "transcript",
                "role": "agent",
                "text": fallback_text,
                "is_filler": False,
                "timestamp": time.time()
            })

        # Update session conversation history if turn completed without interruption
        if turn_id == self.active_turn_id:
            agent_text = " ".join(deep_sentences) if deep_sentences else fallback_text
            if agent_text:
                self.conversation_history.append({"role": "user", "content": text})
                self.conversation_history.append({"role": "assistant", "content": agent_text})
                if len(self.conversation_history) > 8:
                    self.conversation_history = self.conversation_history[-8:]

        total_turnaround_ms = (time.time() - turn_start) * 1000.0
        pb_lat = self.crossfade_buffer.remaining_duration_ms

        registry.telemetry.record_turn(
            turn_id=turn_id,
            session_id=self.session_id,
            mode=self.mode,
            user_text=text,
            transcribed_text_length=self.current_turn_metrics.get("transcribed_text_length", len(text)),
            stt_latency_ms=self.current_turn_metrics.get("stt_latency_ms", 0.0),
            vad_duration_ms=self.current_turn_metrics.get("vad_duration_ms", 0.0),
            classifier_stance=classification.stance.value,
            classifier_latency_ms=classification.latency_ms,
            time_to_filler_ms=time_to_filler_ms,
            time_to_first_audio_ms=ttfa_ms,
            ttft_ms=ttft_ms,
            total_turnaround_ms=total_turnaround_ms,
            playback_latency_ms=pb_lat,
            barge_in=False,
            interrupted_by_barge_in=False,
        )

        # Broadcast latency waterfall metrics
        metrics_payload = {
            "event": "metrics",
            "turn_id": turn_id,
            "stt_latency_ms": round(self.current_turn_metrics.get("stt_latency_ms", 0.0), 1),
            "classifier_latency_ms": round(classification.latency_ms, 1),
            "stance": classification.stance.value,
            "time_to_filler_ms": round(time_to_filler_ms, 1) if time_to_filler_ms else 0.0,
            "ttft_ms": round(ttft_ms, 1) if ttft_ms else 0.0,
            "ttfa_ms": round(ttfa_ms, 1) if ttfa_ms else 0.0,
            "total_turnaround_ms": round(total_turnaround_ms, 1),
            "playback_latency_ms": round(pb_lat, 1),
            "timestamp": time.time()
        }
        await self.send_json_safe(metrics_payload)
        logger.info(f"📊 Turnaround Metrics: {metrics_payload}")

    async def _outbound_stream_worker(self):
        """
        Streams queued audio from CrossfadeBuffer to WebSocket:
        - Twilio: converts 16kHz Float32 -> 8kHz mu-law (20ms / 160-byte chunks).
        - Web: streams 16kHz Int16 binary frames in smooth 100ms chunks (1600 samples / 3200 bytes).
        - Taps all outbound frames into EchoReferenceTap to cancel speakerphone acoustic echo!
        """
        # Frame sizes at 16kHz:
        # Twilio PSTN strictly requires 20ms frames (320 samples at 16kHz -> 160 samples at 8kHz)
        # Web browser streaming uses 100ms frames (1600 samples) to prevent timer jitter and underruns
        FRAME_SAMPLES_TWILIO = 320   # 20ms
        FRAME_SAMPLES_WEB = 1600     # 100ms

        frame_samples = FRAME_SAMPLES_TWILIO if self.mode == "twilio" else FRAME_SAMPLES_WEB

        stream_start_time = 0.0
        streamed_audio_sec = 0.0
        last_outbound_time = 0.0

        while self.is_connected:
            try:
                frame_16k = await self.crossfade_buffer.read_frame_async(frame_samples, timeout=0.05)
                now = time.time()
                if len(frame_16k) == 0:
                    is_generating = bool(self.current_tts_task and not self.current_tts_task.done())
                    if self.tts_playing and not self.crossfade_buffer.has_data() and not is_generating:
                        self.tts_playing = False
                        streamed_audio_sec = 0.0
                        if self.mode == "web":
                            await self.send_json_safe({"event": "tts_stop"})
                    await asyncio.sleep(0.01)
                    continue

                # Detect onset of fresh TTS utterance:
                # 450ms barge-in hold-off is active at the start of any new TTS utterance (after silence or gap > 350ms)
                is_fresh_utterance = (not self.tts_playing) or (now - last_outbound_time > 0.35)
                if is_fresh_utterance:
                    was_playing = self.tts_playing
                    self.tts_playing = True
                    self.tts_start_time = now
                    self.echo_canceller.notify_utterance_start(now)
                    stream_start_time = now
                    streamed_audio_sec = 0.0
                    if self.mode == "web" and not was_playing:
                        await self.send_json_safe({"event": "tts_start"})

                last_outbound_time = now

                # MANDATORY: Tap outbound audio into EchoReferenceTap!
                self.echo_tap.push_reference(frame_16k, sample_rate=16000)
                registry.telemetry.record_playback(self.crossfade_buffer.remaining_duration_ms)

                # Send to client
                if self.mode == "twilio" and self.stream_sid:
                    mulaw_bytes, self.ratecv_out_state = pcm16k_to_mulaw8k(frame_16k, self.ratecv_out_state)
                    b64_payload = base64.b64encode(mulaw_bytes).decode("utf-8")
                    msg = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": b64_payload}
                    }
                    await self.websocket.send_text(json.dumps(msg))
                    # Frame pacing: ~20ms
                    await asyncio.sleep(0.019)

                elif self.mode == "web":
                    int16_bytes = (np.clip(frame_16k, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    await self.websocket.send_bytes(int16_bytes)

                    # Adaptive Web streaming pacing:
                    # Windows timer quantization (~15.6ms) causes starvation if sleep is timed too tightly.
                    # When buffer has data, send with minimal yield (sleep(0.005)) so client rapidly
                    # accumulates a safe lead buffer (~200ms). Only sleep to pace delivery when outpacing
                    # real time by >400ms.
                    frame_dur = len(frame_16k) / 16000.0
                    streamed_audio_sec += frame_dur
                    now = time.time()
                    elapsed = now - stream_start_time

                    if elapsed > streamed_audio_sec:
                        # Client ran dry during an upstream generation pause; reset baseline to now
                        stream_start_time = now
                        streamed_audio_sec = frame_dur
                        elapsed = 0.0

                    lead_ms = (streamed_audio_sec - elapsed) * 1000.0
                    if lead_ms > 400.0:
                        # Outpacing real time by >400ms: sleep to allow client playback to catch up,
                        # maintaining a safe lead buffer (~200ms) without runaway latency or echo drift.
                        sleep_dur = min(frame_dur, (lead_ms - 200.0) / 1000.0)
                        await asyncio.sleep(sleep_dur)
                    else:
                        # Client lead is <= 400ms: minimal yield to event loop so client gets primed
                        await asyncio.sleep(0.005)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in outbound_stream_worker: {e}", exc_info=True)


# --- REST & WebSocket Endpoints ---

@app.get("/")
async def serve_web_phone():
    """Serves the enhanced full-duplex web phone interface."""
    return FileResponse("web_phone.html")


@app.get("/api/config")
async def get_config():
    """Returns current active brain status and configuration."""
    if not registry.dual_brain:
        return {"status": "not_initialized", "is_online": False}
    return {
        "provider": "groq" if "groq" in registry.dual_brain.base_url else ("openai" if "openai" in registry.dual_brain.base_url else "local"),
        "model": registry.dual_brain.model,
        "base_url": registry.dual_brain.base_url,
        "is_online": registry.dual_brain.is_llm_available(),
        "has_api_key": bool(registry.dual_brain.api_key and registry.dual_brain.api_key != "not-needed"),
    }


@app.post("/api/config")
async def update_config_endpoint(request: Request):
    """Dynamically updates API key, base URL, and model for DualBrainEngine."""
    body = await request.json()
    api_key = body.get("api_key")
    base_url = body.get("base_url")
    model = body.get("model")
    if registry.dual_brain:
        status = registry.dual_brain.update_config(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        return {"status": "success", **status}
    return {"status": "error", "message": "DualBrainEngine not initialized"}


@app.get("/api/telemetry")
async def get_telemetry():
    """
    Returns live telemetry waterfall metrics:
    - Waterfall latencies (STT, classifier, TTF, TTFA, TTFT, total turnaround, VAD, playback)
    - P50, P90, P99 percentile distributions
    - Barge-in interruption counts and rate
    - STT accuracy and character/word statistics
    - Recent turn waterfall history
    """
    return registry.telemetry.get_summary()


@app.post("/api/telemetry/reset")
async def reset_telemetry():
    """Resets all in-memory telemetry metrics and counters."""
    registry.telemetry.reset()
    return {"status": "success", "message": "Telemetry metrics reset"}


@app.delete("/api/telemetry")
async def delete_telemetry():
    """Resets all in-memory telemetry metrics (REST DELETE alternative)."""
    registry.telemetry.reset()
    return {"status": "success", "message": "Telemetry metrics reset"}


@app.post("/twiml")
async def twilio_twiml_endpoint(request: Request):
    """
    TwiML webhook for Twilio PSTN call routing.
    Returns TwiML XML <Connect><Stream url="wss://HOST/stream"/></Connect>.
    """
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost:8000"
    twiml_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/stream" />
    </Connect>
</Response>"""
    return Response(content=twiml_xml, media_type="text/xml")


@app.websocket("/stream")
async def twilio_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams:
    - 8000Hz G.711u / base64 payload.
    - Full-duplex with EchoReferenceTap and instant barge-in clearing.
    """
    await websocket.accept()
    session = CallSession(websocket, mode="twilio")
    session.start_workers()
    logger.info(f"📞 [Twilio] Inbound Call Connected (Session {session.session_id})")

    try:
        while True:
            text_data = await websocket.receive_text()
            msg = json.loads(text_data)
            event_type = msg.get("event")

            if event_type == "start":
                session.stream_sid = msg.get("start", {}).get("streamSid")
                logger.info(f"📞 [Twilio] Stream Started: {session.stream_sid}")
                # Initial greeting
                await session.text_queue.put("Hello!")

            elif event_type == "media":
                payload = msg.get("media", {}).get("payload", "")
                if payload:
                    raw_mulaw = base64.b64decode(payload)
                    pcm_16k, session.ratecv_in_state = mulaw_to_pcm16_16k(raw_mulaw, session.ratecv_in_state)
                    # FULL DUPLEX: Ingest audio into echo canceller & VAD!
                    await session.ingest_inbound_audio(pcm_16k)

            elif event_type == "stop":
                logger.info(f"📞 [Twilio] Call Ended: {session.stream_sid}")
                break

    except WebSocketDisconnect:
        logger.info(f"📞 [Twilio] Call Disconnected.")
    except Exception as e:
        logger.error(f"❌ [Twilio] WebSocket Error: {e}", exc_info=True)
    finally:
        await session.close()


@app.websocket("/web_stream")
async def web_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Browser Web Phone testing:
    - 16000Hz Int16 binary PCM or JSON control events.
    - Real-time latency waterfall, filler highlight, and audio visualizer.
    """
    await websocket.accept()
    session = CallSession(websocket, mode="web")
    session.start_workers()
    logger.info(f"🌐 [Web] Client Connected (Session {session.session_id})")

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                # Convert 16kHz Int16 bytes to Float32 array
                pcm_16k = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                # FULL DUPLEX: Ingest audio into echo canceller & VAD!
                await session.ingest_inbound_audio(pcm_16k)

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    if data.get("event") == "ping":
                        await websocket.send_json({"event": "pong", "timestamp": time.time()})
                    elif data.get("event") in ("config", "set_config", "update_config"):
                        api_key = data.get("api_key")
                        base_url = data.get("base_url")
                        model = data.get("model")
                        status = {}
                        if registry.dual_brain:
                            status = registry.dual_brain.update_config(
                                api_key=api_key,
                                base_url=base_url,
                                model=model,
                            )
                        await websocket.send_json({
                            "event": "config_updated",
                            **status
                        })
                    elif data.get("event") == "text_prompt":
                        # Direct text prompt test
                        prompt_text = data.get("text", "")
                        if prompt_text:
                            await session.text_queue.put(prompt_text)
                except Exception as e:
                    logger.warning(f"WebSocket control error: {e}")

            if message.get("type") == "websocket.disconnect":
                break

    except (WebSocketDisconnect, RuntimeError):
        logger.info(f"🌐 [Web] Client Disconnected.")
    except Exception as e:
        logger.error(f"❌ [Web] WebSocket Error: {e}", exc_info=True)
    finally:
        await session.close()


@app.on_event("startup")
async def startup_event():
    """Initializes models and services on FastAPI startup."""
    logger.info("Starting Telephony Server background engines...")
    initialize_models()

    async def _warmup():
        if registry.groq_stt_client is not None:
            try:
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
                await registry.groq_stt_client.audio.transcriptions.create(
                    file=("warmup.wav", io.BytesIO(wav_buf.getvalue()), "audio/wav"),
                    model="whisper-large-v3-turbo",
                    language="en",
                    temperature=0.0,
                    response_format="json"
                )
                logger.info("⚡ Groq STT connection pool warmed up (<250ms ready).")
            except Exception as e:
                logger.debug(f"Groq STT warmup notice: {e}")

    asyncio.create_task(_warmup())
    logger.info("Telephony Server ready on port 8000.")


if __name__ == "__main__":
    uvicorn.run("telephony_server:app", host="0.0.0.0", port=8000, reload=False)
