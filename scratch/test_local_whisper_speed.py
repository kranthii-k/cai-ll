import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from engine.crossfade_buffer import resample_24k_to_16k

kokoro = Kokoro('kokoro-v1.0.onnx', 'voices-v1.0.bin')
samples, _ = kokoro.create('Hello, what are your store hours?', voice='am_adam', speed=1.1, lang='en-us')
pcm_16k = resample_24k_to_16k(samples)

model = WhisperModel('tiny.en', device='cpu', compute_type='int8', cpu_threads=4)
# warmup
model.transcribe(pcm_16k, beam_size=1)

times = []
for _ in range(5):
    t0 = time.time()
    segs, _ = model.transcribe(pcm_16k, beam_size=1)
    text = ' '.join(s.text for s in segs)
    times.append((time.time() - t0) * 1000)

print(f'Local Faster-Whisper tiny.en: min={min(times):.1f}ms, avg={sum(times)/len(times):.1f}ms, text="{text}"')
