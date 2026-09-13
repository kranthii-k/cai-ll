# 🚀 Ultra-Low Latency Voice Pipeline Runbook

This guide outlines the exact sequence to spin up the local voice agent pipeline. The architecture relies on several microservices running concurrently.

## 🛠️ Prerequisites Before You Start
1. **Docker Desktop**: Must be installed and running (Required for LiveKit Server and SIP Bridge).
2. **NVIDIA GPU Setup**: Ensure you have the latest NVIDIA drivers installed. The vLLM server will automatically use CUDA.

---

## 🏃‍♂️ Startup Sequence

You will need to open **4 separate PowerShell terminals** to run these services side-by-side.

### Terminal 1: Boot Telephony Transport (LiveKit)
This starts the WebRTC media server and SIP bridge via Docker.
```powershell
# Run this in the project root
.\start_livekit.ps1
```
*Wait until Docker reports the containers `livekit`, `livekit-sip`, and `redis` are `Running`.*

### Terminal 2: Boot AI LLM Inference (vLLM)
This starts the local LLM OpenAI-compatible server on port `8000`.
```powershell
# Run this in the project root
.\start_vllm.ps1
```
*Note: The first time you run this, it will download a ~15GB Qwen model. Wait until the terminal logs show `Uvicorn running on http://0.0.0.0:8000`.*

### Terminal 3: Boot Voice Agent Worker
This script connects your Python LiveKit worker to the transport server and binds it to the local models.
```powershell
# Run this in the project root
.\start_agent.ps1
```
*Wait for the log: `Connected to room...` or `Agent worker started`.*

### Terminal 4: Boot Outbound Dialer Daemon (Optional)
If you want to trigger outbound phone calls autonomously, start the FastAPI daemon on port `8001`.
```powershell
# Run this in the project root
.\start_daemon.ps1
```

---

## 🧪 How to Test the Pipeline

### Method A: WebRTC Browser Test (Free / No Phone Needed)
1. Go to the [LiveKit Playground (Agents Viewer)](https://agents-playground.livekit.io/)
2. Set the LiveKit URL to: `ws://localhost:7880`
3. Set the API Key to: `devkey`
4. Set the API Secret to: `secret`
5. Click **Connect** and allow microphone permissions. Speak to the agent and test interruptions!

### Method B: Outbound SIP Phone Call
Ensure your `livekit.yaml` and `docker-compose.yml` SIP credentials map to your active Telnyx trunk. 
Trigger a call to your real mobile phone by sending a POST request to your local daemon:
```powershell
Invoke-RestMethod -Uri "http://localhost:8001/dial" -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"number": "+15551234567"}'
```
*Your phone will ring, and the agent will greet you immediately upon picking up!*
