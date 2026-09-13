from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import psutil
from livekit import api
import uuid
import os

app = FastAPI()

class DialRequest(BaseModel):
    number: str
    context: str = ""

@app.post("/dial")
async def dial_out(req: DialRequest):
    # Phase 6: VRAM Load Shedder
    # Mocking VRAM with system RAM for compatibility. 
    # In production, use torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
    mem = psutil.virtual_memory()
    if mem.percent > 85:
        raise HTTPException(status_code=503, detail="Server overloaded (Memory > 85%)")

    livekit_api = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL", "http://localhost:7880"),
        api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
        api_secret=os.getenv("LIVEKIT_API_SECRET", "secret")
    )
    
    room_name = f"room-{uuid.uuid4()}"
    
    # Phase 5: Outbound Daemon
    try:
        await livekit_api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id="YOUR_TRUNK_ID",
                sip_call_to=req.number,
                room_name=room_name,
                participant_identity=f"agent-{req.number}",
            )
        )
    finally:
        await livekit_api.aclose()
        
    return {"status": "dialing", "room": room_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
