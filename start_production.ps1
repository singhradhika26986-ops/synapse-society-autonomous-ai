$Python = "C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not $env:SYNAPSE_ALLOW_MODEL_DOWNLOAD -and -not $env:SYNAPSE_TEXT_MODEL) {
    $env:SYNAPSE_DISABLE_TRANSFORMER = "1"
}
if (-not $env:HOST) { $env:HOST = "127.0.0.1" }
if (-not $env:PORT) { $env:PORT = "8010" }

$NeedsInstall = $false
try {
    & $Python -c "import fastapi, uvicorn, faiss" | Out-Null
} catch {
    $NeedsInstall = $true
}

if ($NeedsInstall) {
    Write-Host "Installing production dependencies..."
    & $Python -m pip install -r requirements-prod.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install production dependencies."
    }
}

& $Python -m uvicorn production_app:app --host $env:HOST --port $env:PORT
