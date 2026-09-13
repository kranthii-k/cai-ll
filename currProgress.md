# Project Progress Tracker

## Current Status: Phase 1 (Local Model Server Validation)

### Phase 1: Local Model Server Validation (No Telephony)
- [x] Initialize Python project and dependencies (`vllm`, `faster-whisper`, `kokoro`, `pysbd`, `silero-vad`)
- [ ] Boot vLLM OpenAI API server on port 8000
- [ ] Test vLLM streaming via `curl` to verify token stream output
- [ ] Run `faster-whisper` and `Kokoro-82M` ONNX runtimes in isolated background processes/threads
- [x] Write `test_local_pipeline.py`: Record laptop mic → Silero VAD → Whisper STT → vLLM token stream → `pysbd` chunker → Kokoro TTS → laptop speaker.
- [ ] Verification Gate: Record timestamp from user speech stop to speaker playback start. Must be <500ms.

### Phase 2: Custom LiveKit Agent Subclasses & WebRTC Test
- [x] Docker compose for self-hosted LiveKit and SIP bridge (`docker-compose.yml`, `livekit.yaml`)
- [x] Create Voice Pipeline Agent (`agent.py`) using local STT, LLM, TTS configurations
- [x] WebRTC browser interrupt testing available via agent deployment

### Phase 3: Telephony Bridge & Firewall Lockdown
- [x] LiveKit SIP docker container configured
- [x] Note: Run `ufw allow 5060` and `ufw allow 10000:20000/udp` on target deployment host.

### Phase 4: Telecom Audio Hardening & Echo Cancellation
- [x] Dynamic VAD Endpointing (800ms/300ms thresholds in `agent.py`)
- [x] Audio denoiser and Echo reference plugins integrated in `requirements.txt` for production

### Phase 5: Agentic Tool Execution & Outbound Dialer
- [x] Tool Integration (`@llm.ai_callable` for Uptime in `agent.py`)
- [x] Outbound Daemon (`daemon.py` FastAPI service for `POST /dial`)

### Phase 6: Production Watchdogs & Safeguards
- [x] VRAM Load Shedder (RAM monitor in `daemon.py`)
- [x] Idle Session Watchdog (180s timeout in `agent.py`)
- [x] Structured Timestamps (Metrics logger in `agent.py`)