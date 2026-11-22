import os
import time
import io
import wave
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sr = 16000
duration = 2.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
audio_float = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
audio_int16 = (audio_float * 32767).astype(np.int16)

wav_buf = io.BytesIO()
with wave.open(wav_buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(audio_int16.tobytes())
wav_bytes = wav_buf.getvalue()

# Test 1: Groq Whisper via OpenAI client
print("--- TEST 1: GROQ WHISPER API VIA OPENAI SDK ---")
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key
    )
    
    for model_name in ["whisper-large-v3-turbo", "distil-whisper-large-v3-en"]:
        try:
            t0 = time.time()
            audio_file = ("test.wav", io.BytesIO(wav_bytes), "audio/wav")
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=model_name,
                language="en",
                response_format="json"
            )
            elapsed = (time.time() - t0) * 1000
            print(f"Groq {model_name} latency: {elapsed:.1f}ms! Result: '{transcription.text}'")
        except Exception as e:
            print(f"Groq {model_name} error: {e}")
else:
    print("No GROQ_API_KEY")

# Test 2: Local Faster-Whisper
print("\n--- TEST 2: LOCAL FASTER-WHISPER ---")
try:
    from faster_whisper import WhisperModel
    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
    for model_size in ["tiny.en", "base.en"]:
        for device, comp in [("cuda", "float16"), ("cpu", "int8")]:
            try:
                t0 = time.time()
                model = WhisperModel(model_size, device=device, compute_type=comp, cpu_threads=4)
                load_time = (time.time() - t0) * 1000
                t0 = time.time()
                segments, _ = model.transcribe(audio_float, beam_size=1)
                text = " ".join([s.text for s in segments])
                infer_time = (time.time() - t0) * 1000
                print(f"Local {model_size} ({device}, {comp}) -> load: {load_time:.0f}ms, infer: {infer_time:.1f}ms")
            except Exception as e:
                print(f"Local {model_size} ({device}, {comp}) failed: {e}")
except Exception as e:
    print(f"Faster-whisper error: {e}")

# Test 3: Kokoro TTS speed
print("\n--- TEST 3: KOKORO TTS SPEED ---")
try:
    from kokoro_onnx import Kokoro
    if os.path.exists("kokoro-v1.0.onnx"):
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        test_sentence = "Hello, how can I assist you today?"
        t0 = time.time()
        samples, sr_out = kokoro.create(test_sentence, voice="af_heart", speed=1.05, lang="en-us")
        elapsed = (time.time() - t0) * 1000
        print(f"Kokoro '{test_sentence}' inference: {elapsed:.1f}ms, duration of audio: {len(samples)/sr_out:.2f}s")
except Exception as e:
    print(f"Kokoro test error: {e}")
