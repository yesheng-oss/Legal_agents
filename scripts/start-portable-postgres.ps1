$ErrorActionPreference = "Stop"

Write-Host "=== Start portable PostgreSQL 16 ===" -ForegroundColor Cyan

$pgRoot = Join-Path $env:USERPROFILE "pgsql16-portable\pgsql"
$dataDir = Join-Path $env:USERPROFILE "pgsql16-portable\data"
$logFile = Join-Path $env:USERPROFILE "pgsql16-portable\postgres.log"
$pgCtl = Join-Path $pgRoot "bin\pg_ctl.exe"
$psql = Join-Path $pgRoot "bin\psql.exe"

if (-not (Test-Path $pgCtl)) {
    Write-Host "PostgreSQL not found: $pgCtl" -ForegroundColor Red
    Write-Host "Expected portable PostgreSQL under: $pgRoot"
    exit 1
}

if (-not (Test-Path $dataDir)) {
    Write-Host "Initializing data directory: $dataDir"
    & (Join-Path $pgRoot "bin\initdb.exe") -D $dataDir -U postgres -A trust -E UTF8 --locale=C
}

$listening = Get-NetTCPConnection -LocalPort 5432 -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    & $pgCtl -D $dataDir -l $logFile -o "-p 5432" start
    Start-Sleep -Seconds 3
}

$listening = Get-NetTCPConnection -LocalPort 5432 -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    Write-Host "PostgreSQL failed to start. Log: $logFile" -ForegroundColor Red
    Get-Content $logFile -Tail 50 -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "PostgreSQL is listening on port 5432." -ForegroundColor Green

$roleSql = "DO `$`$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'legal_agent') THEN CREATE ROLE legal_agent LOGIN PASSWORD 'legal_agent'; ELSE ALTER ROLE legal_agent WITH LOGIN PASSWORD 'legal_agent'; END IF; END `$`$;"
& $psql -U postgres -d postgres -v ON_ERROR_STOP=1 -c $roleSql

$exists = (& $psql -U postgres -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='legal_agent'")
if (($exists | Out-String).Trim() -ne "1") {
    & $psql -U postgres -d postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE legal_agent OWNER legal_agent;"
}

& $psql -U postgres -d legal_agent -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"
& $psql -U postgres -d legal_agent -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','pg_trgm') ORDER BY extname;"

Write-Host "DATABASE_URL=postgresql+psycopg://legal_agent:legal_agent@localhost:5432/legal_agent" -ForegroundColor Green
