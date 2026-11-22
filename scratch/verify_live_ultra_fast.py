import asyncio
import json
import os
import sys
import time
import io
import wave
import numpy as np
import websockets
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

async def test_live_turnaround():
    uri = "ws://127.0.0.1:8000/web_stream"
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as ws:
        print("Connected to /web_stream!")
        
        # Synthesize a real human-like speech phrase using Kokoro if available, or clean speech sample
        # Let's generate a speech audio wav using kokoro for the test input: "Hello, what are your store hours?"
        from kokoro_onnx import Kokoro
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        samples, sr = kokoro.create("Hello, what are your store hours?", voice="am_adam", speed=1.1, lang="en-us")
        
        # Resample from 24000 to 16000
        from engine.crossfade_buffer import resample_24k_to_16k
        speech_16k = resample_24k_to_16k(samples)
        int16_bytes = (speech_16k * 32767.0).astype(np.int16).tobytes()
        
        chunk_size = 1600 * 2  # 100ms frames = 3200 bytes
        print(f"Streaming {len(speech_16k)/16000:.2f}s of real speech to server...")
        
        for i in range(0, len(int16_bytes), chunk_size):
            await ws.send(int16_bytes[i:i+chunk_size])
            await asyncio.sleep(0.08)
            
        # Send 350ms of silence to trigger VAD silence detector
        silence = np.zeros(int(16000 * 0.35), dtype=np.int16).tobytes()
        for i in range(0, len(silence), chunk_size):
            await ws.send(silence[i:i+chunk_size])
            await asyncio.sleep(0.08)
            
        speech_finished_t = time.time()
        print(f"Finished sending speech at t=0.0s. Now measuring turnaround...")
        
        first_audio_t = None
        caller_transcript_t = None
        filler_t = None
        agent_transcript_t = None
        events = []
        audio_frame_count = 0
        total_audio_bytes = 0
        
        t0 = time.time()
        while time.time() - t0 < 8.0:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.5)
                now = time.time() - speech_finished_t
                if isinstance(msg, str):
                    data = json.loads(msg)
                    ev = data.get("event")
                    events.append((now, ev, data))
                    print(f"[{now*1000:6.1f}ms] 📩 JSON: {ev} -> {data.get('text', data.get('stance', ''))}")
                    
                    if ev == "transcript" and data.get("role") == "caller":
                        caller_transcript_t = now
                    elif ev == "filler_start":
                        filler_t = now
                    elif ev == "transcript" and data.get("role") == "agent" and not data.get("is_filler"):
                        agent_transcript_t = now
                elif isinstance(msg, bytes):
                    audio_frame_count += 1
                    total_audio_bytes += len(msg)
                    if first_audio_t is None:
                        first_audio_t = now
                        print(f"[{now*1000:6.1f}ms] 🎵 FIRST AUDIO RECEIVED! Frame size: {len(msg)} bytes")
                    else:
                        if audio_frame_count % 5 == 0:
                            print(f"[{now*1000:6.1f}ms] 🎵 Audio frame #{audio_frame_count} ({len(msg)} bytes)")
            except asyncio.TimeoutError:
                print(f"[{(time.time()-speech_finished_t)*1000:6.1f}ms] Stream idle...")
                if audio_frame_count > 0:
                    break

        print("\n" + "="*50)
        print("⚡ LIVE TURNAROUND PERFORMANCE BENCHMARK ⚡")
        print("="*50)
        print(f"Caller Transcript: {caller_transcript_t*1000:.1f}ms" if caller_transcript_t is not None else "Caller Transcript: None")
        print(f"Fast Filler:       {filler_t*1000:.1f}ms" if filler_t is not None else "Fast Filler: None")
        print(f"First Audio Frame: {first_audio_t*1000:.1f}ms" if first_audio_t is not None else "First Audio: None")
        print(f"Agent Transcript:  {agent_transcript_t*1000:.1f}ms" if agent_transcript_t is not None else "Agent Transcript: None")
        print(f"Total Audio Recv:  {total_audio_bytes} bytes ({total_audio_bytes / 32000:.2f} seconds of speech)")
        print("="*50)

if __name__ == "__main__":
    asyncio.run(test_live_turnaround())
