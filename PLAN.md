```markdown
# SYSTEM SPECIFICATION: ULTRA-LOW LATENCY SELF-HOSTED AGENTIC VOICE PIPELINE
**File Name:** `GOD_MODE_BUILD_SPEC.md`  
**Target Environment:** Dedicated Local/Cloud Host with NVIDIA GPU (24GB VRAM e.g., RTX 4090) or Apple Silicon (32GB+ Unified Memory).  
**Agent Directive:** Read this entire specification before writing code. Do not introduce 3rd-party SaaS AI APIs (No OpenAI, No ElevenLabs, No Deepgram, No Twilio APIs). Re-evaluate and self-correct on every error before proceeding to the next phase.

---

## 1. ARCHITECTURAL BOUNDARIES & STACK

The architecture cleanly decouples **Telephony Transport** from **AI Inference & Agent Logic**.

```text
[ PSTN Phone Network ]
        │
        ▼ (SIP / G.711)
[ Telnyx SIP Trunk ]
        │
        ▼ (Port 5060 TCP/UDP & UDP 10000-20000)
[ LiveKit Server + LiveKit SIP (Docker) ]
        │
        ▼ (WebRTC / Local Inter-Process Transport)
[ Python Agent Worker (livekit-agents) ]
  ├── Acoustic Echo Canceller (livekit-plugins-denoise)
  ├── Silero VAD (ONNX, Local)
  ├── STT: faster-whisper (Local, 16kHz) ➔ Fallback: sherpa-onnx Zipformer
  ├── LLM: vLLM OpenAI-Compatible Server (Llama-3.1-8B / Qwen2.5-7B AWQ)
  │    ├── Agentic Function Calling & Stream Interceptor
  │    └── Sentence Chunker: pysbd
  ├── TTS: Kokoro-82M (ONNX, Local, Sub-50ms)
  └── Outbound Dialer Daemon: FastAPI (Local Port 8000)

```

### Component Matrix

| Layer | Tech Stack | Configuration Constraints |
| --- | --- | --- |
| **Telephony** | Telnyx SIP Trunk | PSTN transport only. G.711 PCMU (u-law) / PCMA (a-law). |
| **Transport Server** | Self-Hosted LiveKit + SIP Bridge | Deployed via Docker Compose. Open UDP ports 10000-20000. |
| **Orchestrator** | `livekit-agents` Python SDK | Custom STT/TTS class overrides. |
| **VAD** | Silero VAD (ONNX) | 20ms frame evaluation, dynamic silence thresholding. |
| **STT** | `faster-whisper` (Tiny/Base) | Audio upsampled from 8kHz SIP to 16kHz PCM. |
| **LLM Engine** | `vLLM` (OpenAI API Server mode) | `Hermes-3-Llama-3.1-8B` or `Qwen2.5-7B-Instruct`. |
| **Sentence Chunker** | `pysbd` | Prevents prosody breaks on "Dr.", "3.14", "p.m.". |
| **TTS Engine** | `Kokoro-82M` (ONNX runtime) | Subclassed into LiveKit Agent stream handler. |
| **AEC / Denoise** | `livekit-plugins-denoise` | `TelephonyDenoiser` + `EchoReferenceTap` tap on outgoing stream. |
| **Outbound Daemon** | FastAPI + `LiveKitApi` | Triggers `sip.create_sip_participant` autonomously. |

---

## 2. HARD RED LINES & NON-NEGOTIABLE CONSTRAINTS

1. **Zero Cloud AI Services:** Do not import or call any external AI APIs. All model weights run on localhost.
2. **VRAM Fencing (vLLM Limit):** vLLM will greedily pre-allocate 90% GPU memory by default. You MUST start vLLM with `--gpu-memory-utilization 0.55` and `--enable-prefix-caching` to reserve VRAM for Whisper, Kokoro, and context scaling.
3. **No Regex Splitting:** Never split LLM streaming tokens into TTS chunks using simple regex punctuation matching. You MUST use `pysbd` to parse valid clause boundaries.
4. **No Raw Un-Muted Tool Calls:** If the LLM generates JSON tool call payloads, the Agent stream processor MUST intercept the initial `{` or `<tool_call>` token and mute the TTS engine so raw JSON is never spoken out loud to the caller.
5. **Mandatory Echo Reference Loop:** The outgoing TTS audio array MUST be tapped via `EchoReferenceTap` and fed into the inbound AEC engine. Failure to do this causes the bot to interrupt itself on speakerphone calls.

---

## 3. PHASED IMPLEMENTATION PLAN

### Phase 1: Local Model Server Validation (No Telephony)

* **Goal:** Confirm local models run concurrently on the GPU without CUDA OOM and achieve sub-500ms TTFA (Time-to-First-Audio).

1. Boot vLLM OpenAI API server on port 8000:
`python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct-AWQ --gpu-memory-utilization 0.55 --enable-prefix-caching`
2. Test vLLM streaming via `curl` to verify token stream output.
3. Run `faster-whisper` and `Kokoro-82M` ONNX runtimes in isolated background processes/threads.
4. Write `test_local_pipeline.py`: Record laptop mic → Silero VAD → Whisper STT → vLLM token stream → `pysbd` chunker → Kokoro TTS → laptop speaker.
5. **Verification Gate:** Record timestamp from user speech stop to speaker playback start. Must be <500ms. If CUDA OOM occurs, reduce vLLM context length or memory utilization.

### Phase 2: Custom LiveKit Agent Subclasses & WebRTC Test

* **Goal:** Plug local models into LiveKit and verify real-time barge-in capability.

1. Spin up self-hosted LiveKit Server via Docker Compose (`livekit.yaml`).
2. Subclass `livekit.agents.stt.STT` to stream incoming audio buffers into your local Whisper instance.
3. Subclass `livekit.agents.tts.TTS` to stream output from Kokoro ONNX as 16kHz audio frames into the LiveKit room. *(Note: Kokoro-FastAPI wrapped endpoints natively bridge well to the LiveKit `openai.TTS` plugin by pointing `base_url` to your localhost Kokoro-FastAPI server)*.
4. Wire the agent using `livekit.agents.pipeline.VoicePipelineAgent`.
5. Connect via LiveKit Playground (Browser).
6. **Verification Gate:** Speak to the agent in the browser. Interrupt the agent mid-sentence. The agent MUST instantly halt TTS playback and pivot to your new utterance.

### Phase 3: Telephony Bridge & Firewall Lockdown

* **Goal:** Enable inbound PSTN calls via Telnyx SIP.

1. Deploy `livekit-sip` bridge via Docker Compose.
2. Configure host firewall (`ufw` or `iptables`): Open TCP/UDP 5060 (SIP) and UDP 10000-20000 (RTP media). Restrict port 5060 inbound traffic to Telnyx IP subnets to prevent SIP brute-force attacks.
3. Map Telnyx SIP trunk to LiveKit dispatch rules.
4. **Verification Gate:** Place a call from a real mobile phone to your Telnyx phone number. Confirm SIP handshake completes and the LiveKit Python agent answers the call.

### Phase 4: Telecom Audio Hardening & Echo Cancellation

* **Goal:** Handle 8kHz PSTN audio degradation and speakerphone echo loops.

1. Enable `TelephonyDenoiser` on the inbound stream.
2. Attach `EchoReferenceTap` to the outbound TTS pipeline.
3. Implement **Dynamic VAD Endpointing**:
* If user pauses but sentence is grammatically incomplete (or contains filler words "um", "ah"): Set VAD silence timeout to **800ms**.
* If sentence is a completed thought: Set VAD silence timeout to **300ms**.


4. Filter Whisper hallucinations: Drop transcription results if `no_speech_prob > 0.6`.
5. **Verification Gate:** Put your mobile phone on speakerphone during a live call. Confirm the agent does NOT cut itself off or trigger false barge-in events from its own speaker audio.

### Phase 5: Agentic Tool Execution & Outbound Dialer

* **Goal:** Give the agent agency to execute code and autonomously initiate outgoing calls.

1. **Tool Integration:** Register Python methods using `@llm.ai_callable`. Pass system schemas to vLLM.
2. Intercept vLLM token streams: If a tool call pattern is detected, buffer tokens, execute the Python function async, feed the returned JSON back to vLLM, and generate the final spoken response.
3. **Outbound Daemon:** Implement a FastAPI service exposing `POST /dial`:
```python
@app.post("/dial")
async def dial_out(number: str, context: str):
    await livekit_api.sip.create_sip_participant(
        CreateSIPParticipantRequest(
            sip_trunk_id="YOUR_TRUNK_ID",
            sip_call_to=number,
            room_name=f"room-{uuid4()}",
            participant_identity=f"agent-{number}"
        )
    )

```


4. Inject `context` into the Agent's system prompt upon room entry.
5. **Verification Gate:**
* Inbound: Ask the agent a question requiring a tool call (e.g., "Check system uptime"). Confirm tool runs and response is spoken without reading raw JSON.
* Outbound: Trigger `POST /dial`. Confirm the system autonomously dials your mobile phone and opens the conversation upon pick-up.



### Phase 6: Production Watchdogs & Safeguards

* **Goal:** Ensure 24/7 stability without zombie processes or memory leaks.

1. **VRAM Load Shedder:** Monitor GPU memory before accepting room connections. If VRAM exceeds 85%, reject incoming SIP invites with `486 Busy Here`.
2. **Idle Session Watchdog:** If Silero VAD detects 180 seconds of total silence, drop the LiveKit room session and issue a SIP hangup to reclaim GPU memory.
3. **Structured Timestamps:** Log structured JSON events for every conversation turn:
`{"event": "turn_metrics", "vad_end_ms": 120, "stt_ms": 110, "llm_ttft_ms": 85, "tts_first_chunk_ms": 35, "total_ttfa_ms": 350}`

---

## 4. TROUBLESHOOTING & RE-EVALUATION PROTOCOLS

If an error occurs during build or execution, apply these deterministic fixes:

* **Error: `CUDA out of memory` during vLLM boot or TTS synthesis**
* *Fix:* Check vLLM memory utilization parameter. Reduce `--gpu-memory-utilization` from 0.55 to 0.45. Ensure Kokoro is running in ONNX runtime mode rather than PyTorch GPU allocation mode.


* **Error: Loud screeching or robot voice on mobile phone**
* *Fix:* Sample rate mismatch. PSTN audio is 8kHz G.711. Ensure your audio transport layer explicitly upsamples to 16kHz PCM (S16LE) before feeding Silero VAD and Whisper. Ensure Kokoro downsamples 24kHz/16kHz to 8kHz when returning frames to LiveKit SIP.


* **Error: Agent gets stuck in an infinite interruption loop**
* *Fix:* Echo Cancellation is failing. Adjust `EchoReferenceTap` buffer delay alignment to match Telnyx SIP trunk latency (typically 40ms–120ms). Increase Silero VAD `min_speech_duration_ms` threshold from 50ms to 150ms.


* **Error: SIP calls drop immediately after 30 seconds**
* *Fix:* NAT / SDP firewall issue. Ensure your router/firewall is forwarding UDP ports 10000-20000 and that LiveKit `RTC_NODE_IP` is configured with your public server IP address.



---

## 5. FINAL ACCEPTANCE CHECKLIST

* [ ] vLLM, Whisper, and Kokoro run 100% locally with zero external API calls.
* [ ] Time-to-First-Audio (TTFA) measured and logged under 500ms.
* [ ] Speakerphone test passed without self-interruption loops.
* [ ] Tool execution works in real-time without speaking JSON strings.
* [ ] Outbound SIP dialer API successfully triggers external calls.
* [ ] Firewall port rules active and watchdog timers running.

```

```