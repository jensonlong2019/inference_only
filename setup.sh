#!/usr/bin/env bash
# GPT-SoVITS 推理功能 - 一键环境安装脚本 (Linux / macOS / WSL)
# 用法: bash setup.sh

set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "  GPT-SoVITS 推理 - 一键环境安装"
echo "=========================================="
echo "工作目录: $(pwd)"
echo ""

# 1. 检查 Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "未检测到 Python，请先安装 Python 3.10 或 3.11。"
    exit 1
fi
PYTHON=$(command -v python3 2>/dev/null || command -v python)
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/5] Python: $($PYTHON --version)"

if ! command -v ffmpeg &>/dev/null; then
    echo "  ⚠ 未检测到 ffmpeg（合成音频必需）"
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "    无 brew 可运行: bash install_ffmpeg_mac.sh"
        echo "    有 brew 可运行: brew install ffmpeg"
    else
        echo "    安装命令: sudo apt install ffmpeg"
    fi
fi
if [[ "$PY_VER" == "3.9" ]]; then
    echo "  提示: Python 3.9 可用，但建议 3.10/3.11。"
elif [[ "$PY_VER" < "3.9" ]]; then
    echo "  错误: 需要 Python 3.9+"
    exit 1
fi
echo ""

# 2. 创建虚拟环境
if [ ! -f "venv/bin/activate" ]; then
    echo "[2/5] 创建虚拟环境 venv ..."
    $PYTHON -m venv venv
else
    echo "[2/5] 已存在 venv，跳过创建。"
fi
echo ""

source venv/bin/activate
pip install --upgrade pip

# 3. 安装 torch（固定 2.3.1，避免 2.6+ weights_only 等兼容问题）
echo "[3/5] 安装 torch / torchaudio (2.3.1) ..."
pip install --no-cache-dir torch==2.3.1 torchaudio==2.3.1

# 4. 安装其余依赖 + 版本锁定
echo "[4/5] 安装项目依赖 ..."
pip install -r requirements.txt

echo "锁定兼容版本 ..."
pip install --upgrade --force-reinstall \
    "numpy<2" \
    "huggingface_hub<1.0" \
    "markupsafe~=2.1.5" \
    "transformers<4.46.0" \
    "fastapi==0.112.4" \
    "starlette==0.38.2" \
    "pydantic<2.10" \
    "urllib3<2"

# macOS 系统 Python 链的 LibreSSL 警告
if [[ "$(uname -s)" == "Darwin" ]]; then
    pip install "urllib3<2" 2>/dev/null || true
fi

# LangSegment：PyPI 0.2.0 有 bug
echo "安装 LangSegment ..."
python install_langsegment.py

# 5. 验证
echo ""
echo "[5/5] 环境验证 ..."
python check_env.py --quick

echo ""
echo "=========================================="
echo "  安装完成"
echo "=========================================="
echo "启动: ./start_inference.sh"
echo "排错: python check_env.py"
echo "=========================================="
