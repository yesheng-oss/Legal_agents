Write-Host "=== 法律 RAG 知识库系统 ===" -ForegroundColor Cyan
Write-Host ""

$step = 0

# Step 1: Check Ollama
$step++
Write-Host "[$step/4] 检查 Ollama 服务..." -ForegroundColor Yellow
try {
    $ollamaOk = $false
    try { $ollamaOk = (ollama list 2>$null) -match "qwen2.5" } catch {}
    if (-not $ollamaOk) {
        Write-Host "  -> Pulling qwen2.5:7b (首次运行需要下载 4.7GB 模型)..."
        ollama pull qwen2.5:7b
    }
    else { Write-Host "  -> qwen2.5:7b 已就绪" -ForegroundColor Green }
}
catch { Write-Host "  -> Ollama 未运行，请先启动 ollama serve" -ForegroundColor Red }

# Step 2: Download data
$step++
Write-Host "[$step/4] 下载法律数据..." -ForegroundColor Yellow
python src/download.py
if ($LASTEXITCODE -ne 0) { Write-Host "下载失败，请检查网络" -ForegroundColor Red; exit 1 }

# Step 3: Build index
$step++
Write-Host "[$step/4] 构建向量索引..." -ForegroundColor Yellow
python src/ingest.py
if ($LASTEXITCODE -ne 0) { Write-Host "索引构建失败" -ForegroundColor Red; exit 1 }

# Step 4: Start query
$step++
Write-Host "[$step/4] 启动查询界面..." -ForegroundColor Yellow
python src/rag.py
