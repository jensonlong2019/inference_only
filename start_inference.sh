#!/bin/bash
# GPT-SoVITS 推理功能启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 优先使用项目自带的 ffmpeg（install_ffmpeg_mac.sh 安装，无需 brew）
if [ -x "$SCRIPT_DIR/bin/ffmpeg" ]; then
    export PATH="$SCRIPT_DIR/bin:$PATH"
fi

# 设置 Python 路径，确保能找到 GPT_SoVITS 子模块（feature_extractor 等）
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/GPT_SoVITS:$PYTHONPATH"

# 设置默认端口（可通过环境变量覆盖）
export infer_ttswebui=${infer_ttswebui:-9872}

# 设置语言（可通过环境变量覆盖）
LANGUAGE=${language:-Auto}

echo "=========================================="
echo "GPT-SoVITS 推理功能启动"
echo "=========================================="
echo "工作目录: $SCRIPT_DIR"
echo "端口: $infer_ttswebui"
echo "语言: $LANGUAGE"
echo "=========================================="

# 检查 Python 环境
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "错误: 未找到 Python，请先安装 Python 3.7+"
    exit 1
fi

# 检查并激活虚拟环境（按优先级顺序）
VENV_ACTIVATED=false

# 1. 检查当前目录下的 venv
if [ -d "venv" ]; then
    echo "检测到虚拟环境 (当前目录)，正在激活..."
    source venv/bin/activate
    VENV_ACTIVATED=true
# 2. 检查项目同级目录下的 venv（GPT-SoVITS-v2-2025/venv）
elif [ -d "../GPT-SoVITS-v2-2025/venv" ]; then
    echo "检测到虚拟环境 (项目目录)，正在激活..."
    source ../GPT-SoVITS-v2-2025/venv/bin/activate
    VENV_ACTIVATED=true
# 3. 检查上级目录的 venv
elif [ -d "../venv" ]; then
    echo "检测到虚拟环境 (上级目录)，正在激活..."
    source ../venv/bin/activate
    VENV_ACTIVATED=true
fi

# 如果未激活虚拟环境，检查是否已安装 torch
if [ "$VENV_ACTIVATED" = false ]; then
    echo "未找到虚拟环境，检查系统 Python 环境..."
    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        echo "✓ 检测到已安装的 torch，使用系统 Python 环境"
    else
        echo ""
        echo "⚠️  警告: 未找到虚拟环境，且系统 Python 未安装 torch"
        echo ""
        echo "请选择以下方案之一："
        echo "1. 创建虚拟环境并安装依赖："
        echo "   python3 -m venv venv"
        echo "   source venv/bin/activate"
        echo "   pip install -r requirements.txt"
        echo ""
        echo "2. 或使用项目目录的虚拟环境："
        echo "   将 inference_only 放回 GPT-SoVITS-v2-2025 目录下"
        echo ""
        echo "3. 或在系统 Python 中安装依赖："
        echo "   pip install -r requirements.txt"
        echo ""
        read -p "是否继续尝试启动？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# 启动前快速环境检查
if [ -f "check_env.py" ]; then
    if ! $PYTHON_CMD check_env.py --quick 2>/dev/null; then
        echo ""
        echo "环境检查未通过，请先运行: bash setup.sh"
        echo "或查看详情: python check_env.py"
        exit 1
    fi
fi

# 创建必要的目录
mkdir -p output/batch_result
mkdir -p GPT_weights_v2
mkdir -p GPT_weights
mkdir -p SoVITS_weights_v2
mkdir -p SoVITS_weights

# 启动推理服务
echo "正在启动推理服务..."
$PYTHON_CMD GPT_SoVITS/inference_webui.py "$LANGUAGE"

