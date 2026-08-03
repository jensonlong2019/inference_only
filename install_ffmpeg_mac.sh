#!/usr/bin/env bash
# macOS 无需 brew，下载 ffmpeg 到项目 bin/ 目录
# 用法: bash install_ffmpeg_mac.sh

set -e
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "此脚本仅适用于 macOS。"
    exit 1
fi

if command -v ffmpeg &>/dev/null; then
    echo "系统已安装 ffmpeg: $(command -v ffmpeg)"
    ffmpeg -version | head -1
    exit 0
fi

mkdir -p bin
ARCH=$(uname -m)
case "$ARCH" in
    arm64)
        URL="https://github.com/eugeneware/ffmpeg-static/releases/download/b6.0/ffmpeg-darwin-arm64"
        ;;
    x86_64)
        URL="https://github.com/eugeneware/ffmpeg-static/releases/download/b6.0/ffmpeg-darwin-x64"
        ;;
    *)
        echo "不支持的架构: $ARCH"
        exit 1
        ;;
esac

echo "正在下载 ffmpeg ($ARCH) ..."
echo "来源: $URL"
if command -v curl &>/dev/null; then
    curl -fL --progress-bar "$URL" -o bin/ffmpeg
elif command -v wget &>/dev/null; then
    wget -O bin/ffmpeg "$URL"
else
    echo "需要 curl 或 wget 来下载。"
    exit 1
fi

chmod +x bin/ffmpeg
xattr -d com.apple.quarantine bin/ffmpeg 2>/dev/null || true

echo ""
echo "安装完成:"
bin/ffmpeg -version | head -1
echo ""
echo "ffmpeg 路径: $(pwd)/bin/ffmpeg"
echo "启动服务时会自动使用，直接运行: ./start_inference.sh"
