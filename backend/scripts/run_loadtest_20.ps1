# 密码仅在当前进程环境中临时传入Locust，不写入脚本、日志或CSV。
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$trialRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $trialRoot
# 只验证Python与Locust启动，不读取密码、不发送任何压测请求。
if ($CheckOnly) {
    & "$trialRoot\.venv\Scripts\python.exe" -m locust --version
    if ($LASTEXITCODE -ne 0) { throw "Locust startup check failed: $LASTEXITCODE" }
    Write-Host 'PASS: Locust launcher is available; no load test started.'
    return
}
$previousPassword = $env:LOCUST_PASSWORD
try {
    $securePassword = Read-Host 'Password for loadtest001-loadtest020' -AsSecureString
    $env:LOCUST_PASSWORD = ([System.Net.NetworkCredential]::new('', $securePassword)).Password
    $env:LOCUST_ACCOUNT_COUNT = '20'
    $env:LOCUST_ACCOUNT_START = '1'
    $env:LOCUST_MODE = 'all'
    $env:LOCUST_RAG_PER_USER = '1'
    $env:LOCUST_SSE_DEADLINE = '0'
    $env:LOCUST_STOP_AFTER_SAMPLES = '1'
    $trialStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $trialPrefix = "loadtest_20_single_b2_$trialStamp"
    $env:LOCUST_METRICS_FILE = "${trialPrefix}_sse_metrics.csv"
    Write-Host "20 users | 1 RAG per user | no duration cutoff | output=$trialPrefix"
    # 不设置-t；20个请求全部成功或失败并记录后，由脚本自动停止。
    & "$trialRoot\.venv\Scripts\python.exe" -m locust -f locustfile.py --host http://127.0.0.1:8000 --headless -u 20 -r 1 --csv $trialPrefix
    $trialExitCode = $LASTEXITCODE
    Write-Host "Locust finished with exit code: $trialExitCode"
    if ($trialExitCode -ne 0) {
        Write-Warning 'Locust failed; inspect the errors above and the generated CSV files.'
    }
}
finally {
    $env:LOCUST_PASSWORD = $previousPassword
    if ($securePassword) { $securePassword.Dispose() }
}
exit $trialExitCode
