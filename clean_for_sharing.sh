#!/bin/bash
# 清理脚本：移除不必要的文件，准备分享

echo "正在清理 inference_only 文件夹..."

# 删除 Python 缓存文件
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

# 删除用户生成的文件
rm -rf output/batch_result/*.mp3
rm -rf output/batch_result/*.wav

# 删除用户配置文件（可选，保留空文件）
# rm -f user_cache.json
# rm -f weight.json

# 删除日志文件
find . -type f -name "*.log" -delete 2>/dev/null

# 删除临时文件
find . -type f -name "*.tmp" -delete 2>/dev/null
find . -type f -name ".DS_Store" -delete 2>/dev/null

echo "✓ 清理完成！"
echo ""
echo "现在可以压缩 inference_only 文件夹分享了。"
echo "压缩命令: zip -r GPT-SoVITS-Inference.zip inference_only/"

