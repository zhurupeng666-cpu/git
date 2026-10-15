# 前台启动本轮单Worker服务；关闭本窗口或Ctrl+C即可停止，不另起隐藏实例。
$ErrorActionPreference = 'Stop'
$trialRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $trialRoot
$env:LOCAL_INFERENCE_THREADS = '2'
$env:LOCAL_INFERENCE_CONCURRENCY = '1'
$env:RERANK_BATCH_SIZE = '2'
$env:RERANK_MAX_CANDIDATES = '20'
$env:QUERY_VECTOR_CACHE_SIZE = '512'
$env:SSE_TIMEOUT_SECONDS = '0'
$env:LLM_REQUEST_TIMEOUT_SECONDS = '0'
$env:INTENT_REQUEST_TIMEOUT_SECONDS = '0'
$env:TOOL_QUERY_TIMEOUT_MS = '0'
$env:RAG_QUERY_TIMEOUT_MS = '0'
Write-Host 'Single worker | inference threads=2 | rerank batch=2 | business timeout=off'
& "$trialRoot\.venv\Scripts\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
exit $LASTEXITCODE
