import asyncio
import time
import queue
import numpy as np
import sounddevice as sd
from collections import deque
import torch
import os
import urllib.request
import site

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

# --- App State for Thread Communication ---
app_state = {
    "tts_playing": False
}

# Setup constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
VAD_THRESHOLD = 0.1  # Low threshold for quiet mics
SILENCE_TIMEOUT_MS = 300  # HYPER-OPTIMIZED: Reduced to 300ms (absolute minimum human pause).

# Queues for pipeline
audio_queue = asyncio.Queue()
text_queue = asyncio.Queue()
speech_queue = asyncio.Queue()

# --- Download Kokoro models if missing ---
if not os.path.exists("kokoro-v1.0.onnx"):
    print("Downloading Kokoro ONNX model (this may take a minute)...")
    urllib.request.urlretrieve("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx", "kokoro-v1.0.onnx")
if not os.path.exists("voices-v1.0.bin"):
    print("Downloading Kokoro voices...")
    urllib.request.urlretrieve("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin", "voices-v1.0.bin")

print("Loading Kokoro TTS...")
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# --- Initialize Models ---
print("Loading VAD model...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

print("Loading Whisper Model (Base EN optimized with CPU threads for speed/accuracy)...")
whisper_model = WhisperModel("base.en", device="cpu", compute_type="default", cpu_threads=6)

print("Loading OpenAI Client (Ollama)...")
client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

segmenter = pysbd.Segmenter(language="en", clean=False)

def get_audio_callback(loop):
    def audio_callback(indata, frames, time_info, status):
        """Callback for sounddevice InputStream"""
        if app_state["tts_playing"]:
            return  # Drop audio to prevent the AI from hearing itself (feedback loop)
            
        if status:
            pass
        
        # Windows mics often have >1 channels (e.g. 2 or 4). 
        # DO NOT average them (it causes phase cancellation on mic arrays). Just take the primary channel.
        if indata.shape[1] > 1:
            mono_data = indata[:, 0:1]
        else:
            mono_data = indata
            
        # VERY IMPORTANT: This callback runs on a separate C-thread. 
        # We MUST use call_soon_threadsafe to push to the asyncio Queue!
        loop.call_soon_threadsafe(audio_queue.put_nowait, mono_data.astype('float32').copy())
    return audio_callback

async def mic_listener():
    """Listens to mic and applies VAD to chunk speech"""
    print(f"Starting mic listener using Default Device... Speak into your microphone!")
    loop = asyncio.get_running_loop()
    
    # Use channels=None so Windows doesn't hang trying to force 4-channels into 1 natively
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=None, callback=get_audio_callback(loop), blocksize=CHUNK_SIZE, dtype='float32')
    
    with stream:
        audio_buffer = []
        pre_speech_buffer = deque(maxlen=15)  # Stores ~0.5s of audio to catch the first syllable
        is_speaking = False
        silence_start = None
        chunk_count = 0
        print("DEBUG: Audio stream successfully opened.")

        while True:
            chunk = await audio_queue.get()
            chunk_tensor = torch.from_numpy(chunk.flatten())
            
            chunk_count += 1
            if chunk_count % 30 == 0 and not is_speaking:
                print(f"[DEBUG] Current mic volume: {np.max(np.abs(chunk)):.4f} (Threshold to speak: {VAD_THRESHOLD})", end='\r', flush=True)
            
            speech_prob = vad_model(chunk_tensor, SAMPLE_RATE).item()
            
            if speech_prob > VAD_THRESHOLD:
                if not is_speaking:
                    print("\n[VAD] Speech detected!")
                    is_speaking = True
                    audio_buffer.extend(pre_speech_buffer)  # Add history so we don't chop off the start
                audio_buffer.append(chunk.flatten())
                silence_start = None
            else:
                if not is_speaking:
                    pre_speech_buffer.append(chunk.flatten())  # Maintain rolling history
                else:
                    audio_buffer.append(chunk.flatten())  # IMPORTANT: Keep recording during the silence pause!
                    if silence_start is None:
                        silence_start = time.time()
                    elif (time.time() - silence_start) * 1000 > SILENCE_TIMEOUT_MS:
                        print("[VAD] Speech ended. Processing...")
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
        
        # beam_size=1 disables beam search for speed. vad_filter=True prevents hallucinations.
        segments, info = whisper_model.transcribe(
            audio_data, 
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False
        )
        text = " ".join([segment.text for segment in segments]).strip()
        
        stt_time = (time.time() - start_time) * 1000
        if text:
            print(f"🗣️ You: {text} (STT time: {stt_time:.2f}ms)")
            await text_queue.put(text)

async def llm_tts_worker():
    """Sends text to LLM, chunks with pysbd, and plays via TTS"""
    print("LLM/TTS Worker ready.")
    
    while True:
        user_text = await text_queue.get()
        start_time = time.time()
        
        try:
            response = await client.chat.completions.create(
                model="qwen2.5:0.5b",
                messages=[
                    {"role": "system", "content": "You are a helpful, ultra-low latency voice assistant. Keep answers brief and conversational. DO NOT use emojis or formatting."},
                    {"role": "user", "content": user_text}
                ],
                stream=True
            )
            
            buffer = ""
            first_token_time = None
            
            async for chunk in response:
                if not first_token_time:
                    first_token_time = time.time()
                    ttft = (first_token_time - start_time) * 1000
                    print(f"⏱️ [vLLM] Time-to-First-Token: {ttft:.2f}ms")
                    
                content = chunk.choices[0].delta.content
                if content:
                    buffer += content
                    
                    # Split sentences
                    sentences = segmenter.segment(buffer)
                    if len(sentences) > 1 or (buffer.strip() and buffer.strip()[-1] in ".!?"):
                        sentence = sentences[0]
                        
                        # Only speak if it contains actual words (skip stray "." or "?")
                        if any(c.isalnum() for c in sentence):
                            print(f"🤖 Agent: {sentence.strip()}")
                            
                            app_state["tts_playing"] = True
                            try:
                                # Generate and play TTS synchronously for simplicity in this script
                                samples, sample_rate = await asyncio.to_thread(kokoro.create, sentence, voice="af_heart", speed=1.0, lang="en-us")
                                await asyncio.to_thread(sd.play, samples, sample_rate)
                                await asyncio.to_thread(sd.wait)
                            finally:
                                app_state["tts_playing"] = False
                        
                        buffer = buffer[len(sentence):].lstrip()
                        
            # Play any remaining buffer
            remaining = buffer.strip()
            if remaining and any(c.isalnum() for c in remaining):
                print(f"🤖 Agent: {remaining}")
                
                app_state["tts_playing"] = True
                try:
                    samples, sample_rate = await asyncio.to_thread(kokoro.create, remaining, voice="af_heart", speed=1.0, lang="en-us")
                    await asyncio.to_thread(sd.play, samples, sample_rate)
                    await asyncio.to_thread(sd.wait)
                finally:
                    app_state["tts_playing"] = False
                
        except Exception as e:
            print(f"LLM Error: Make sure Ollama is running! {e}")

async def main():
    await asyncio.gather(
        mic_listener(),
        stt_worker(),
        llm_tts_worker()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
