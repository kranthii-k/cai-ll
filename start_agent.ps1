.\venv\Scripts\activate
$env:LIVEKIT_URL="ws://localhost:7880"
$env:LIVEKIT_API_KEY="devkey"
$env:LIVEKIT_API_SECRET="secret"
python agent.py start
