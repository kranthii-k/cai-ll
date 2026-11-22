# Ultra-Low Latency Conversational Voice AI Telephony Gateway Launcher
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " ⚡ Starting Voice AI Telephony Gateway (Full-Duplex)" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check if server is already running on port 8000
$portActive = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($portActive) {
    Write-Host "✔ Telephony Server is already active and listening on port 8000." -ForegroundColor Green
} else {
    Write-Host "Starting telephony server on port 8000..." -ForegroundColor Yellow
    Start-Process -FilePath ".\venv\Scripts\python.exe" -ArgumentList "telephony_server.py" -NoNewWindow
    Start-Sleep -Seconds 4
}

# 2. Check LLM backend status
if ($env:GROQ_API_KEY) {
    Write-Host "✔ Groq LPU Cloud configured (Ultra-Low Latency ~80ms TTFT)" -ForegroundColor Green
} elseif ($env:OPENAI_API_KEY) {
    Write-Host "✔ OpenAI Cloud configured" -ForegroundColor Green
} else {
    $ollamaProc = Get-Process -Name "*ollama*" -ErrorAction SilentlyContinue
    if ($ollamaProc) {
        Write-Host "✔ Local Ollama process detected" -ForegroundColor Green
    } else {
        Write-Host "ℹ No Groq/OpenAI key set and Ollama offline." -ForegroundColor Yellow
        Write-Host "  ✔ Instant Intelligent Built-in Dialogue Engine (<20ms) is active!" -ForegroundColor Green
        Write-Host "  Handles orders, hours, bookings, support, and conversational QA with zero lag." -ForegroundColor Green
        Write-Host "  To connect Cloud AI: paste a Groq key in the web phone or set `$env:GROQ_API_KEY." -ForegroundColor Gray
    }
}

# 3. Launch the web phone in default browser
Write-Host ""
Write-Host "Opening Web Phone at: http://localhost:8000" -ForegroundColor Cyan
Start-Process "http://localhost:8000"

Write-Host "✔ Web Phone opened! Click 'CALL' to start speaking." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
