import os
import io
import site
import wave
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import Response, JSONResponse
import uvicorn
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
import asyncio
from pydantic import BaseModel

# CUDA DLL FIX FOR WINDOWS
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

app = FastAPI(title="Local Audio API")

print("Loading Whisper...")
whisper_model = WhisperModel("base.en", device="cpu", compute_type="default", cpu_threads=6)

print("Loading Kokoro...")
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

@app.post("/v1/audio/transcriptions")
async def create_transcription(file: UploadFile = File(...), model: str = Form("whisper-1")):
    """Mock OpenAI STT Endpoint"""
    audio_bytes = await file.read()
    
    # Write to temp file because faster_whisper expects file path or binary stream
    temp_io = io.BytesIO(audio_bytes)
    
    # Transcribe
    segments, info = whisper_model.transcribe(temp_io, beam_size=1, vad_filter=False)
    text = " ".join([segment.text for segment in segments]).strip()
    
    return JSONResponse(content={"text": text})

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = "alloy"

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    """Mock OpenAI TTS Endpoint"""
    text = req.input
    
    # Generate audio
    samples, sample_rate = await asyncio.to_thread(kokoro.create, text, voice="af_heart", speed=1.0, lang="en-us")
    
    # Convert float32 to int16 PCM
    pcm_16 = (samples * 32767).astype(np.int16)
    
    # Write WAV file in memory
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_16.tobytes())
        
    wav_io.seek(0)
    return Response(content=wav_io.read(), media_type="audio/wav")

if __name__ == "__main__":
    # We run on 8080 (or 8000) depending on config. We'll run on 8080.
    uvicorn.run(app, host="0.0.0.0", port=8080)
