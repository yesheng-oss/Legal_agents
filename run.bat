@echo off
chcp 65001 >nul
echo === 法律 RAG 知识库系统 ===
echo.

echo [1/4] 检查 Ollama 服务...
ollama list 2>nul | findstr "qwen2.5" >nul
if %ERRORLEVEL% neq 0 (
    echo   -^> Pulling qwen2.5:7b (首次运行需要下载 4.7GB 模型)...
    ollama pull qwen2.5:7b
) else (
    echo   -^> qwen2.5:7b 已就绪
)

echo [2/4] 下载法律数据...
python src\download.py
if %ERRORLEVEL% neq 0 (
    echo 下载失败，请检查网络
    pause
    exit /b 1
)

echo [3/4] 构建向量索引...
python src\ingest.py
if %ERRORLEVEL% neq 0 (
    echo 索引构建失败
    pause
    exit /b 1
)

echo [4/4] 启动查询界面...
python src\rag.py
pause
