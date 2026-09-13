import asyncio
import time
import json
import base64
import numpy as np
import torch
import os
import urllib.request
import site
import audioop
from collections import deque
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

# FIX FOR WINDOWS CUDA DLL ERROR (cublas64_12.dll not found)
if os.name == 'nt':
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
    try:
        site_packages = site.getsitepackages()
        for sp in site_packages:
            cublas_bin = os.path.join(sp, "nvidia", "cublas", "bin")
            cudnn_bin = os.path.join(sp, "nvidia", "cudnn", "bin")
            if os.path.exists(cublas_bin):
                os.add_dll_directory(cublas_bin)
            if os.path.exists(cudnn_bin):
                os.add_dll_directory(cudnn_bin)
    except Exception:
        pass

from faster_whisper import WhisperModel
import pysbd
from openai import AsyncOpenAI
from kokoro_onnx import Kokoro

# --- App State ---
app_state = {
    "tts_playing": False,
    "active_ws": None,
    "stream_sid": None,
    "mode": None # 'twilio' or 'web'
}

# Setup constants
SAMPLE_RATE_TWILIO = 8000
SAMPLE_RATE_VAD = 16000
SAMPLE_RATE_TTS = 24000
VAD_THRESHOLD = 0.1
SILENCE_TIMEOUT_MS = 300

# Queues for pipeline
audio_queue = asyncio.Queue()
text_queue = asyncio.Queue()
speech_queue = asyncio.Queue()

# --- Initialize Models ---
print("Loading Kokoro TTS...")
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

print("Loading VAD model...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

print("Loading Whisper Model (Base EN on CPU for stability)...")
whisper_model = WhisperModel("base.en", device="cpu", compute_type="default", cpu_threads=6)

print("Loading OpenAI Client (Ollama)...")
client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

segmenter = pysbd.Segmenter(language="en", clean=False)

app = FastAPI()

async def vad_worker():
    """Processes audio chunks from the websocket, applies VAD, and builds speech chunks"""
    audio_buffer = []
    pre_speech_buffer = deque(maxlen=30)  # Stores ~1s of 16kHz audio to catch the first syllable
    is_speaking = False
    silence_start = None

    while True:
        chunk = await audio_queue.get()
        chunk_tensor = torch.from_numpy(chunk)
        
        # We process VAD in 512 sample increments (32ms at 16kHz) to keep it accurate
        # If chunk is larger, we split it
        for i in range(0, len(chunk_tensor), 512):
            sub_chunk = chunk_tensor[i:i+512]
            if len(sub_chunk) < 512:
                # pad with zeros if necessary
                sub_chunk = torch.nn.functional.pad(sub_chunk, (0, 512 - len(sub_chunk)))
                
            speech_prob = vad_model(sub_chunk, SAMPLE_RATE_VAD).item()
            
            if speech_prob > VAD_THRESHOLD:
                if not is_speaking:
                    print("\n[VAD] Phone speech detected!")
                    is_speaking = True
                    audio_buffer.extend(pre_speech_buffer)
                audio_buffer.append(sub_chunk.numpy())
                silence_start = None
            else:
                if not is_speaking:
                    pre_speech_buffer.append(sub_chunk.numpy())
                else:
                    audio_buffer.append(sub_chunk.numpy())
                    if silence_start is None:
                        silence_start = time.time()
                    elif (time.time() - silence_start) * 1000 > SILENCE_TIMEOUT_MS:
                        print("[VAD] Phone speech ended. Transcribing...")
                        is_speaking = False
                        audio_data = np.concatenate(audio_buffer)
                        await speech_queue.put(audio_data)
                        audio_buffer = []
                        pre_speech_buffer.clear()

async def stt_worker():
    """Processes speech chunks through Whisper"""
    while True:
        audio_data = await speech_queue.get()
        start_time = time.time()
        
        segments, info = whisper_model.transcribe(
            audio_data, 
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False
        )
        text = " ".join([segment.text for segment in segments]).strip()
        
        stt_time = (time.time() - start_time) * 1000
        if text:
            print(f"🗣️ Caller: {text} (STT time: {stt_time:.2f}ms)")
            await text_queue.put(text)

async def send_audio_to_client(ws: WebSocket, text: str):
    """Generates TTS and streams it to Twilio or the Web Client"""
    app_state["tts_playing"] = True
    try:
        if app_state["mode"] == "web":
            if ws == app_state["active_ws"]:
                await ws.send_json({"event": "tts_start"})
                
        # Generate Float32 audio at 24000Hz
        samples, sample_rate = await asyncio.to_thread(kokoro.create, text, voice="af_heart", speed=1.0, lang="en-us")
        
        # Convert to int16 PCM
        pcm_16_24k = (samples * 32767).astype(np.int16).tobytes()
        
        if app_state["mode"] == "twilio":
            # Twilio needs 8000Hz mu-law base64
            pcm_16_8k, _ = audioop.ratecv(pcm_16_24k, 2, 1, 24000, 8000, None)
            mulaw_out = audioop.lin2ulaw(pcm_16_8k, 2)
            
            chunk_size = 160 
            for i in range(0, len(mulaw_out), chunk_size):
                chunk = mulaw_out[i:i+chunk_size]
                b64_payload = base64.b64encode(chunk).decode("utf-8")
                message = {
                    "event": "media",
                    "streamSid": app_state["stream_sid"],
                    "media": {"payload": b64_payload}
                }
                if ws == app_state["active_ws"]:
                    await ws.send_json(message)
                    await asyncio.sleep(0.02)
                    
        elif app_state["mode"] == "web":
            # Web client can just take the raw 24000Hz int16 bytes!
            chunk_size = 4096
            for i in range(0, len(pcm_16_24k), chunk_size):
                chunk = pcm_16_24k[i:i+chunk_size]
                if ws == app_state["active_ws"]:
                    await ws.send_bytes(chunk)
                    # Rough sleep to stream in real-time
                    await asyncio.sleep(chunk_size / (24000 * 2))
                    
    except Exception as e:
        print(f"Error streaming audio: {e}")
    finally:
        if app_state["mode"] == "web" and ws == app_state["active_ws"]:
            try:
                await ws.send_json({"event": "tts_stop"})
            except:
                pass
        app_state["tts_playing"] = False

async def llm_tts_worker():
    """Sends text to LLM, chunks with pysbd, and streams TTS to phone"""
    print("LLM/TTS Worker ready.")
    
    while True:
        user_text = await text_queue.get()
        start_time = time.time()
        
        try:
            response = await client.chat.completions.create(
                model="qwen2.5:0.5b",
                messages=[
                    {"role": "system", "content": "You are a helpful, ultra-low latency telephone voice assistant. Keep answers brief, conversational, and completely free of emojis or formatting."},
                    {"role": "user", "content": user_text}
                ],
                stream=True
            )
            
            buffer = ""
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    buffer += content
                    
                    sentences = segmenter.segment(buffer)
                    if len(sentences) > 1 or (buffer.strip() and buffer.strip()[-1] in ".!?"):
                        sentence = sentences[0]
                        if any(c.isalnum() for c in sentence):
                            print(f"🤖 Agent: {sentence.strip()}")
                            if app_state["active_ws"]:
                                await send_audio_to_client(app_state["active_ws"], sentence.strip())
                        buffer = buffer[len(sentence):].lstrip()
                        
            remaining = buffer.strip()
            if remaining and any(c.isalnum() for c in remaining):
                print(f"🤖 Agent: {remaining}")
                if app_state["active_ws"]:
                    await send_audio_to_client(app_state["active_ws"], remaining)
                
        except Exception as e:
            print(f"LLM Error: {e}")

@app.get("/")
async def serve_web_phone():
    return FileResponse("web_phone.html")

@app.websocket("/web_stream")
async def web_websocket_endpoint(websocket: WebSocket):
    """Handles raw audio streaming directly from the web browser"""
    await websocket.accept()
    print("\n[Web-Phone] Call Connected!")
    app_state["active_ws"] = websocket
    app_state["mode"] = "web"
    
    try:
        while True:
            # Receive raw Int16 binary audio from the browser (16000Hz)
            data = await websocket.receive_bytes()
            
            if app_state["tts_playing"]:
                continue
                
            # Convert Int16 bytes to Float32 numpy array
            np_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            await audio_queue.put(np_audio)
            
    except WebSocketDisconnect:
        print("[Web-Phone] Call Disconnected.")
        app_state["active_ws"] = None
    except Exception as e:
        print(f"[Web-Phone] Error: {e}")
        app_state["active_ws"] = None

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Handles incoming Twilio WebSocket connection"""
    await websocket.accept()
    print("\n[Twilio] Call Connected!")
    app_state["active_ws"] = websocket
    app_state["mode"] = "twilio"
    
    # State for resampling
    cv_state = None
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg['event'] == 'start':
                app_state["stream_sid"] = msg['start']['streamSid']
                print(f"[Twilio] Stream Started: {app_state['stream_sid']}")
                
                # Optional: Send a greeting when they pick up
                await text_queue.put("Hello!")
                
            elif msg['event'] == 'media':
                # If AI is currently speaking, we drop incoming audio to prevent hearing itself!
                if app_state["tts_playing"]:
                    continue
                    
                # Decode base64 mulaw payload
                payload = msg['media']['payload']
                raw_mulaw = base64.b64decode(payload)
                
                # Convert mu-law to 16-bit PCM (8000 Hz)
                pcm_16 = audioop.ulaw2lin(raw_mulaw, 2)
                
                # Resample 8000 Hz to 16000 Hz for VAD and Whisper
                pcm_16k, cv_state = audioop.ratecv(pcm_16, 2, 1, 8000, 16000, cv_state)
                
                # Convert to float32 numpy array
                np_audio = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0
                
                await audio_queue.put(np_audio)
                
            elif msg['event'] == 'stop':
                print("[Twilio] Call Disconnected.")
                app_state["active_ws"] = None
                break
    except Exception as e:
        print(f"[Twilio] Connection Error: {e}")
        app_state["active_ws"] = None

@app.post("/twiml")
async def twilio_webhook(request: Request):
    """Endpoint for Twilio to configure the call"""
    # Dynamically grab the Ngrok host to construct the WSS URL
    host = request.headers.get("host", "localhost:8000")
    
    return HTMLResponse(content=f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/stream" />
    </Connect>
</Response>""", media_type="text/xml")

@app.on_event("startup")
async def startup_event():
    # Start the background workers
    asyncio.create_task(vad_worker())
    asyncio.create_task(stt_worker())
    asyncio.create_task(llm_tts_worker())
    print("\nTelephony Server running on port 8000")
    print("Use Ngrok to expose this port: ngrok http 8000")

if __name__ == "__main__":
    uvicorn.run("telephony_pipeline:app", host="0.0.0.0", port=8000, reload=False)
