import asyncio
import os
import sys
import time
import io
import wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import AsyncOpenAI
from kokoro_onnx import Kokoro
from engine.crossfade_buffer import resample_24k_to_16k
from dotenv import load_dotenv

load_dotenv()

async def test_acc():
    kokoro = Kokoro('kokoro-v1.0.onnx', 'voices-v1.0.bin')
    samples, _ = kokoro.create('Hello, what are your store hours?', voice='am_adam', speed=1.1, lang='en-us')
    pcm_16k = resample_24k_to_16k(samples)
    audio_int16 = (pcm_16k * 32767.0).astype(np.int16)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_int16.tobytes())
    wav_bytes = wav_buf.getvalue()

    client = AsyncOpenAI(base_url='https://api.groq.com/openai/v1', api_key=os.getenv('GROQ_API_KEY'))
    # warmup
    await client.audio.transcriptions.create(
        file=('a.wav', io.BytesIO(wav_bytes), 'audio/wav'),
        model='whisper-large-v3-turbo',
        language='en',
        response_format='json'
    )
    
    t0 = time.time()
    res = await client.audio.transcriptions.create(
        file=('a.wav', io.BytesIO(wav_bytes), 'audio/wav'),
        model='whisper-large-v3-turbo',
        language='en',
        response_format='json'
    )
    dur = (time.time() - t0) * 1000
    print(f'Transcription: "{res.text}" in {dur:.1f}ms')

if __name__ == '__main__':
    asyncio.run(test_acc())
