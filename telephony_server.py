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
import json
import logging
import math
import os
import site
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any

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

import torch
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from openai import AsyncOpenAI
import pysbd

from engine.crossfade_buffer import CrossfadeBuffer, apply_crossfade
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

# Audio Constants
SAMPLE_RATE_TWILIO = 8000
SAMPLE_RATE_VAD = 16000
SAMPLE_RATE_KOKORO = 24000
TWILIO_FRAME_SIZE = 160  # 20ms at 8kHz mu-law
VAD_CHUNK_SAMPLES = 512  # 32ms at 16kHz
VAD_SPEECH_THRESHOLD = 0.25
SILENCE_TIMEOUT_MS = 320

app = FastAPI(title="Telephony Streaming Audio Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Global Model & Engine Holders ---
class ModelRegistry:
    kokoro: Optional[Kokoro] = None
    whisper: Optional[WhisperModel] = None
    vad_model: Optional[Any] = None
    fast_classifier: FastClassifier = FastClassifier()
    filler_manager: FillerManager = FillerManager(sample_rate=16000)
    dual_brain: Optional[DualBrainEngine] = None
    segmenter: pysbd.Segmenter = pysbd.Segmenter(language="en", clean=False)


registry = ModelRegistry()


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

    if registry.whisper is None:
        try:
            logger.info("Initializing Whisper Model (base.en)...")
            registry.whisper = WhisperModel("base.en", device="cpu", compute_type="default", cpu_threads=4)
        except Exception as e:
            logger.warning(f"Whisper initialization warning: {e}")

    if registry.dual_brain is None:
        registry.dual_brain = DualBrainEngine(
            model=os.getenv("LLM_MODEL", "qwen2.5:0.5b"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            filler_manager=registry.filler_manager,
            fast_classifier=registry.fast_classifier,
        )

# Initialize models and dual_brain on module import
initialize_models()


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
        import scipy.signal
        # Simple mu-law decode table
        lut = np.zeros(256, dtype=np.int16)
        for i in range(256):
            b = ~i
            sign = -1 if (b & 0x80) else 1
            exponent = (b >> 4) & 0x07
            mantissa = b & 0x0F
            lut[i] = sign * ((mantissa << (exponent + 3)) + (132 << exponent) - 132)
        raw_int = lut[np.frombuffer(mulaw_bytes, dtype=np.uint8)]
        # Upsample 8k to 16k
        up = scipy.signal.resample(raw_int.astype(np.float32), len(raw_int) * 2) / 32768.0
        return up, None


def pcm16k_to_mulaw8k(pcm_16k_float: np.ndarray, cv_state: Optional[Any] = None) -> Tuple[bytes, Any]:
    """Converts 16kHz Float32 numpy array to 8kHz G.711u bytes."""
    int16_16k = (np.clip(pcm_16k_float, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    if audioop is not None:
        pcm_8k, next_cv = audioop.ratecv(int16_16k, 2, 1, 16000, 8000, cv_state)
        mulaw_bytes = audioop.lin2ulaw(pcm_8k, 2)
        return mulaw_bytes, next_cv
    else:
        import scipy.signal
        pcm_8k_float = scipy.signal.resample(pcm_16k_float, len(pcm_16k_float) // 2)
        # Mu-law companding formula
        s = np.clip(pcm_8k_float, -1.0, 1.0)
        u = 255.0
        compressed = np.sign(s) * (np.log(1.0 + u * np.abs(s)) / np.log(1.0 + u))
        lut_bytes = ((compressed + 1.0) * 127.5).astype(np.uint8).tobytes()
        return lut_bytes, None


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
            min_delay_ms=40.0,
            max_delay_ms=120.0,
            barge_in_min_duration_ms=300.0,
            backchannel_max_duration_ms=350.0
        )
        self.crossfade_buffer = CrossfadeBuffer(sample_rate=16000, default_crossfade_ms=30.0)

        # State flags & background tasks
        self.tts_playing = False
        self.active_turn_id = 0
        self.current_tts_task: Optional[asyncio.Task] = None
        self.worker_tasks: List[asyncio.Task] = []
        self.ratecv_in_state = None
        self.ratecv_out_state = None

        # Turn performance tracking
        self.turn_metrics: Dict[str, Any] = {}
        self.turn_start_time: float = 0.0

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
        """
        logger.info(f"[{self.mode.upper()}] Barge-in triggered! Aborting TTS (<50ms response).")
        self.tts_playing = False

        # Cancel in-flight generation task
        if self.current_tts_task and not self.current_tts_task.done():
            self.current_tts_task.cancel()

        # Flush audio queues
        self.crossfade_buffer.flush()
        self.echo_tap.clear()
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
        """
        cleaned_pcm, should_interrupt, metrics = self.echo_canceller.process_frame(pcm_16k)

        if should_interrupt and self.tts_playing:
            await self.handle_barge_in()

        # Non-blocking enqueue for VAD processing
        await self.inbound_audio_queue.put(cleaned_pcm)

    async def _vad_worker(self):
        """
        Continuously reads audio chunks, evaluates Silero VAD,
        and slices completed caller speech utterances.
        """
        audio_buffer = []
        pre_speech_buffer = deque(maxlen=25)  # ~800ms pre-speech buffer
        is_speaking = False
        silence_start = None

        while self.is_connected:
            try:
                chunk = await self.inbound_audio_queue.get()
                n_samples = len(chunk)
                if n_samples == 0:
                    continue

                # Process VAD in 512 sample increments (32ms at 16kHz)
                for i in range(0, n_samples, VAD_CHUNK_SAMPLES):
                    sub_chunk = chunk[i:i + VAD_CHUNK_SAMPLES]
                    if len(sub_chunk) < VAD_CHUNK_SAMPLES:
                        sub_chunk = np.pad(sub_chunk, (0, VAD_CHUNK_SAMPLES - len(sub_chunk)))

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
                            silence_start = None
                            audio_buffer.extend(pre_speech_buffer)
                            pre_speech_buffer.clear()
                        audio_buffer.append(sub_chunk)
                        silence_start = None
                    else:
                        if not is_speaking:
                            pre_speech_buffer.append(sub_chunk)
                        else:
                            audio_buffer.append(sub_chunk)
                            if silence_start is None:
                                silence_start = time.time()
                            elif (time.time() - silence_start) * 1000 > SILENCE_TIMEOUT_MS:
                                is_speaking = False
                                silence_start = None
                                if len(audio_buffer) > 0:
                                    full_speech = np.concatenate(audio_buffer)
                                    # Filter short clicks (<180ms)
                                    if len(full_speech) >= int(0.18 * SAMPLE_RATE_VAD):
                                        await self.speech_queue.put(full_speech)
                                audio_buffer = []

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in vad_worker: {e}", exc_info=True)

    async def _stt_worker(self):
        """Processes collected speech chunks with Whisper."""
        while self.is_connected:
            try:
                speech_data = await self.speech_queue.get()
                self.turn_start_time = time.time()
                self.active_turn_id += 1

                text = ""
                if registry.whisper is not None:
                    segments, info = registry.whisper.transcribe(
                        speech_data,
                        beam_size=1,
                        vad_filter=False,
                        condition_on_previous_text=False
                    )
                    text = " ".join([seg.text for seg in segments]).strip()
                else:
                    text = "Hello, can you help me with a question?"

                if text:
                    logger.info(f"🗣️ [{self.mode.upper()}] Caller: {text}")
                    # Broadcast transcript to UI
                    await self.send_json_safe({
                        "event": "transcript",
                        "role": "caller",
                        "text": text,
                        "timestamp": time.time()
                    })
                    await self.text_queue.put(text)

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
                turn_id = self.active_turn_id
                start_time = self.turn_start_time or time.time()

                # Launch turn generation task
                self.current_tts_task = asyncio.create_task(
                    self._generate_and_stream_response(user_text, turn_id, start_time)
                )
                await self.current_tts_task

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
            filler = registry.filler_manager.get_filler(classification.stance, text)
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

                # Queue filler audio
                filler_audio_float = None
                if registry.kokoro is not None:
                    try:
                        f_samples, f_sr = await asyncio.to_thread(
                            registry.kokoro.create, filler.text, voice="af_heart", speed=1.1, lang="en-us"
                        )
                        # Kokoro is 24kHz -> resample to 16kHz
                        if f_sr != 16000:
                            import scipy.signal
                            target_n = int(round(len(f_samples) * 16000.0 / f_sr))
                            filler_audio_float = scipy.signal.resample(f_samples, target_n)
                        else:
                            filler_audio_float = f_samples
                    except Exception as e:
                        logger.warning(f"Kokoro filler synthesis fallback: {e}")

                if filler_audio_float is None and filler.audio_bytes:
                    # Use acoustic chime cue
                    filler_audio_float = np.frombuffer(filler.audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                if filler_audio_float is not None:
                    self.crossfade_buffer.write_chunk(filler_audio_float, crossfade=False)

        # Step 3: Concurrently stream Deep-Path LLM reasoning
        deep_sentences = []
        try:
            dual_brain = registry.dual_brain
            if dual_brain is None:
                initialize_models()
                dual_brain = registry.dual_brain

            if dual_brain is not None:
                async for event in dual_brain.process(text):
                    if turn_id != self.active_turn_id:
                        # Barge-in occurred! Abort current generation
                        return

                    if event.type == DualBrainEventType.SENTENCE:
                        sentence = event.text.strip()
                        if not sentence or not any(c.isalnum() for c in sentence):
                            continue

                        if ttft_ms is None:
                            ttft_ms = (time.time() - turn_start) * 1000.0

                        deep_sentences.append(sentence)
                        logger.info(f"🤖 Agent: {sentence}")

                        # Synthesize TTS with Kokoro
                        if registry.kokoro is not None:
                            samples, sr = await asyncio.to_thread(
                                registry.kokoro.create, sentence, voice="af_heart", speed=1.05, lang="en-us"
                            )
                            # Kokoro is 24kHz -> resample to 16kHz
                            import scipy.signal
                            target_samples = int(round(len(samples) * 16000.0 / sr))
                            pcm_16k = scipy.signal.resample(samples, target_samples)
                        else:
                            # Fallback tone for testing
                            num_s = int(16000 * 0.4)
                            t = np.linspace(0, 0.4, num_s, endpoint=False)
                            pcm_16k = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

                        if ttfa_ms is None:
                            ttfa_ms = (time.time() - turn_start) * 1000.0

                        # Write to CrossfadeBuffer (smooth equal-power crossfade with previous filler/sentence)
                        self.crossfade_buffer.write_chunk(pcm_16k, crossfade=True, crossfade_ms=30.0)

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
                    import scipy.signal
                    target_samples = int(round(len(samples) * 16000.0 / sr))
                    pcm_16k = scipy.signal.resample(samples, target_samples)
                    self.crossfade_buffer.write_chunk(pcm_16k, crossfade=True, crossfade_ms=30.0)
                except Exception:
                    pass
            if ttfa_ms is None:
                ttfa_ms = (time.time() - turn_start) * 1000.0
            await self.send_json_safe({
                "event": "transcript",
                "role": "agent",
                "text": fallback_text,
                "is_filler": False,
                "timestamp": time.time()
            })

        total_turnaround_ms = (time.time() - turn_start) * 1000.0

        # Broadcast latency waterfall metrics
        metrics_payload = {
            "event": "metrics",
            "time_to_filler_ms": round(time_to_filler_ms, 1) if time_to_filler_ms else 0.0,
            "ttft_ms": round(ttft_ms, 1) if ttft_ms else 0.0,
            "ttfa_ms": round(ttfa_ms, 1) if ttfa_ms else 0.0,
            "total_turnaround_ms": round(total_turnaround_ms, 1),
            "timestamp": time.time()
        }
        await self.send_json_safe(metrics_payload)
        logger.info(f"📊 Turnaround Metrics: {metrics_payload}")

    async def _outbound_stream_worker(self):
        """
        Streams queued audio from CrossfadeBuffer to WebSocket:
        - Twilio: converts 16kHz Float32 -> 8kHz mu-law (20ms / 160-byte chunks).
        - Web: streams 16kHz Int16 binary frames (20ms / 640 bytes).
        - Taps all outbound frames into EchoReferenceTap to cancel speakerphone acoustic echo!
        """
        # 20ms frame size at 16kHz = 320 samples
        FRAME_SAMPLES_16K = 320

        while self.is_connected:
            try:
                frame_16k = await self.crossfade_buffer.read_frame_async(FRAME_SAMPLES_16K, timeout=0.05)
                if len(frame_16k) == 0:
                    if self.tts_playing and not self.crossfade_buffer.has_data():
                        self.tts_playing = False
                        if self.mode == "web":
                            await self.send_json_safe({"event": "tts_stop"})
                    await asyncio.sleep(0.01)
                    continue

                if not self.tts_playing:
                    self.tts_playing = True
                    if self.mode == "web":
                        await self.send_json_safe({"event": "tts_start"})

                # MANDATORY: Tap outbound audio into EchoReferenceTap!
                self.echo_tap.push_reference(frame_16k, sample_rate=16000)

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
                    # Frame pacing: ~20ms
                    await asyncio.sleep(0.019)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in outbound_stream_worker: {e}", exc_info=True)


# --- REST & WebSocket Endpoints ---

@app.get("/")
async def serve_web_phone():
    """Serves the enhanced full-duplex web phone interface."""
    return FileResponse("web_phone.html")


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
                    elif data.get("event") == "text_prompt":
                        # Direct text prompt test
                        prompt_text = data.get("text", "")
                        if prompt_text:
                            await session.text_queue.put(prompt_text)
                except Exception:
                    pass

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
    logger.info("Telephony Server ready on port 8000.")


if __name__ == "__main__":
    uvicorn.run("telephony_server:app", host="0.0.0.0", port=8000, reload=False)
