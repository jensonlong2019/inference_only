#!/bin/bash

# 设置颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 检查 ffmpeg 是否安装
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}错误: 未找到 ffmpeg，请先安装 ffmpeg (例如: brew install ffmpeg 或 apt install ffmpeg)${NC}"
    exit 1
fi

# 检查 output 目录
OUTPUT_ROOT="./output"
if [ ! -d "$OUTPUT_ROOT" ]; then
    # 尝试在当前目录寻找 output
    OUTPUT_ROOT=$(find . -maxdepth 2 -type d -name "output" | head -n 1)
    if [ -z "$OUTPUT_ROOT" ]; then
        echo -e "${RED}错误: 未找到 output 文件夹。请确保在项目根目录下运行此脚本。${NC}"
        exit 1
    fi
fi

# 函数：交互式选择目录
select_directory() {
    local current_dir=$1
    local dirs=()
    
    # 获取当前目录下的子目录列表 (排除已优化的文件夹)
    while IFS= read -r d; do
        if [[ "$d" != *"_已优化" ]]; then
            dirs+=("$d")
        fi
    done < <(find "$current_dir" -maxdepth 1 -type d ! -path "$current_dir")

    echo -e "\n${BLUE}--- 当前位置: $current_dir ---${NC}"
    echo "0) [确定选择当前文件夹]"
    
    if [ ${#dirs[@]} -gt 0 ]; then
        for i in "${!dirs[@]}"; do
            echo "$((i+1))) 进入子文件夹: ${dirs[$i]##*/}"
        done
    else
        echo -e "${YELLOW}(此目录下没有更多子文件夹)${NC}"
    fi
    
    echo "b) [返回上一层]"
    echo "q) [退出脚本]"

    read -p "请输入编号: " choice

    case "$choice" in
        0)
            SELECTED_DIR="$current_dir"
            return 0
            ;;
        q)
            echo "已退出。"
            exit 0
            ;;
        b)
            if [ "$current_dir" == "$OUTPUT_ROOT" ] || [ "$current_dir" == "." ]; then
                echo -e "${YELLOW}已经是根目录了。${NC}"
                select_directory "$current_dir"
            else
                select_directory "$(dirname "$current_dir")"
            fi
            return $?
            ;;
        *)
            if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -le "${#dirs[@]}" ] && [ "$choice" -gt 0 ]; then
                select_directory "${dirs[$((choice-1))]}"
                return $?
            else
                echo -e "${RED}无效选择，请重试。${NC}"
                select_directory "$current_dir"
                return $?
            fi
            ;;
    esac
}

# 开始选择
SELECTED_DIR=""
select_directory "$OUTPUT_ROOT"

if [ -z "$SELECTED_DIR" ]; then
    exit 0
fi

# 如果选中的是 output 根目录本身，需要特殊处理以防名称太奇怪
if [ "$SELECTED_DIR" == "./output" ] || [ "$SELECTED_DIR" == "output" ]; then
    TARGET_DIR="output_已优化"
else
    TARGET_DIR="${SELECTED_DIR}_已优化"
fi

echo -e "\n${GREEN}已选中目标: $SELECTED_DIR${NC}"
echo -e "${GREEN}优化结果将保存至: $TARGET_DIR${NC}"

# 创建目标文件夹
mkdir -p "$TARGET_DIR"

# 统计信息
count=0
echo -e "${BLUE}开始处理...${NC}"

# 寻找所有音频文件
# 使用 -print0 处理带空格的文件名
while IFS= read -r -d '' src_file; do
    # 计算相对路径，保持子目录结构
    rel_path="${src_file#$SELECTED_DIR/}"
    target_file="$TARGET_DIR/${rel_path%.*}.mp3"
    
    # 创建子目录结构
    mkdir -p "$(dirname "$target_file")"
    
    # 优化参数说明:
    # -acodec libmp3lame: 使用 mp3 编码
    # -b:a 64k: 码率设为 64k (对于语音来说足够清晰且体积很小)
    # -ar 44100: 采样率 44.1kHz
    # -ac 1: 转为单声道 (体积减半)
    # -loglevel error: 只显示错误信息
    
    ffmpeg -y -i "$src_file" -acodec libmp3lame -b:a 64k -ar 44100 -ac 1 "$target_file" -loglevel error
    
    if [ $? -eq 0 ]; then
        echo -e "完成: ${BLUE}$rel_path${NC}"
        ((count++))
    else
        echo -e "${RED}失败: $rel_path${NC}"
    fi
done < <(find "$SELECTED_DIR" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) -print0)

echo -e "\n${GREEN}全部处理完成！共优化 $count 个文件。${NC}"
echo -e "结果目录: ${YELLOW}$TARGET_DIR${NC}"

# 显示体积对比
if command -v du &> /dev/null; then
    orig_size=$(du -sh "$SELECTED_DIR" | cut -f1)
    new_size=$(du -sh "$TARGET_DIR" | cut -f1)
    echo -e "总大小对比: ${RED}$orig_size${NC} -> ${GREEN}$new_size${NC}"
fi

# 打包结果目录为 zip（与结果目录同级，名称为: <结果目录名>.zip）
if command -v zip &> /dev/null; then
    target_parent_dir="$(cd "$(dirname "$TARGET_DIR")" && pwd)"
    target_basename="$(basename "$TARGET_DIR")"
    zip_path="$target_parent_dir/${target_basename}.zip"

    # 覆盖旧 zip（避免 zip 交互询问）
    if [ -f "$zip_path" ]; then
        rm -f "$zip_path"
    fi

    (
        cd "$target_parent_dir" || exit 1
        zip -r -q "${target_basename}.zip" "$target_basename"
    )

    if [ $? -eq 0 ]; then
        echo -e "最后打包出一个：${GREEN}${target_basename}.zip${NC}"
    else
        echo -e "${RED}打包失败: ${zip_path}${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}未找到 zip 命令，跳过打包步骤（macOS 通常自带 zip）。${NC}"
fi
