$Python = "C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not $env:SYNAPSE_ALLOW_MODEL_DOWNLOAD -and -not $env:SYNAPSE_TEXT_MODEL) {
    $env:SYNAPSE_DISABLE_TRANSFORMER = "1"
}
& $Python "$PSScriptRoot\main.py" @args
