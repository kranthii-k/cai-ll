import asyncio
import websockets
import json
import time
import numpy as np

async def test_ws_latency():
    uri = "ws://127.0.0.1:8000/web_stream"
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        print("Connected!")
        
        # Synthesize 1.5 seconds of simulated voice audio (sine wave 300Hz at 16kHz)
        sr = 16000
        dur = 1.5
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        int16_bytes = (audio * 32767.0).astype(np.int16).tobytes()
        
        # Send audio in 100ms chunks to simulate user speaking
        chunk_size = 1600 * 2 # 1600 samples * 2 bytes = 3200 bytes
        print("Sending 1.5s of speech...")
        speech_start = time.time()
        for i in range(0, len(int16_bytes), chunk_size):
            await ws.send(int16_bytes[i:i+chunk_size])
            await asyncio.sleep(0.05)
            
        print("Finished sending speech. Now sending 0.5s of silence...")
        silence = np.zeros(8000, dtype=np.int16).tobytes()
        for i in range(0, len(silence), chunk_size):
            await ws.send(silence[i:i+chunk_size])
            await asyncio.sleep(0.05)
            
        speech_end = time.time()
        print(f"Speech finished at t=0.0s")
        
        t0 = time.time()
        first_audio_time = None
        first_agent_transcript = None
        
        # Collect events for 15 seconds
        while time.time() - t0 < 15:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                elapsed = time.time() - t0
                if isinstance(msg, str):
                    data = json.loads(msg)
                    ev = data.get("event")
                    print(f"[{elapsed*1000:6.1f}ms] JSON Event: {ev} -> {data}")
                    if ev == "transcript" and data.get("role") == "agent":
                        if first_agent_transcript is None:
                            first_agent_transcript = elapsed
                elif isinstance(msg, bytes):
                    if first_audio_time is None:
                        first_audio_time = elapsed
                        print(f"[{elapsed*1000:6.1f}ms] 🎵 FIRST BINARY AUDIO RECEIVED! ({len(msg)} bytes)")
                    else:
                        print(f"[{elapsed*1000:6.1f}ms] 🎵 Binary Audio chunk ({len(msg)} bytes)")
            except asyncio.TimeoutError:
                print(f"[{(time.time()-t0)*1000:6.1f}ms] Timeout waiting for message...")
                if first_audio_time is not None:
                    break

        print("\n=== SUMMARY ===")
        print(f"First Agent Transcript: {first_agent_transcript*1000 if first_agent_transcript else 'NONE'} ms")
        print(f"First Audio Frame:     {first_audio_time*1000 if first_audio_time else 'NONE'} ms")
        if first_agent_transcript and first_audio_time:
            print(f"DELTA (Audio - Transcript): {(first_audio_time - first_agent_transcript)*1000:.1f} ms")

if __name__ == "__main__":
    asyncio.run(test_ws_latency())
