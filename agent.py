import asyncio
import logging
import time
import json
import os
from typing import AsyncIterable
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, silero

logger = logging.getLogger("agent")

# Phase 5: Agentic Tool Execution
class SystemUptimeTool(llm.AgentFunctionContext):
    @llm.ai_callable(description="Check the system uptime.")
    async def check_uptime(self):
        logger.info("Executing Tool: SystemUptimeTool")
        return "The system has been up for 42 hours."

async def entrypoint(ctx: JobContext):
    # Phase 6: Idle Session Watchdog
    start_time = time.time()
    last_speech_time = time.time()

    async def idle_watchdog():
        while True:
            await asyncio.sleep(5)
            if time.time() - last_speech_time > 180:
                logger.warning("Session idle for >180s. Reclaiming GPU memory.")
                await ctx.room.disconnect()
                break

    asyncio.create_task(idle_watchdog())

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"Connected to room: {ctx.room.name}")
    
    # Phase 2: Configure Local Models
    # Local vLLM
    local_llm = openai.LLM(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        api_key="not-needed"
    )
    
    # Local Kokoro TTS (mapped via OpenAI-compatible endpoint)
    local_tts = openai.TTS(
        base_url="http://localhost:8080/v1",
        model="kokoro",
        api_key="not-needed"
    )
    
    # Local Faster-Whisper (mapped via OpenAI-compatible endpoint for integration)
    local_stt = openai.STT(
        base_url="http://localhost:8000/v1",
        model="whisper-1",
        api_key="not-needed"
    )
    
    fctx = SystemUptimeTool()
    
    # Pipeline Agent
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=local_stt,
        llm=local_llm,
        tts=local_tts,
        fctx=fctx,
        # Phase 4: Dynamic VAD Endpoints
        min_endpointing_delay=0.3,
        max_endpointing_delay=0.8
    )

    # Phase 6: Structured Timestamps
    @agent.on("metrics_collected")
    def on_metrics(metrics):
        structured_log = {
            "event": "turn_metrics",
            "llm_ttft_ms": getattr(metrics, 'llm_ttft_ms', 0),
            "tts_first_chunk_ms": getattr(metrics, 'tts_ttfa_ms', 0),
            "total_ttfa_ms": getattr(metrics, 'total_ttfa_ms', 0)
        }
        logger.info(json.dumps(structured_log))
        
    @agent.on("user_speech_committed")
    def on_user_speech(msg):
        nonlocal last_speech_time
        last_speech_time = time.time()
        
    agent.start(ctx.room)
    
    await asyncio.sleep(1)
    await agent.say("Hello, I am the low latency agent. How can I help you?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=None,
        )
    )
