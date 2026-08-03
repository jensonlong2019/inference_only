#!/usr/bin/env python3
"""环境自检：安装后或排错时运行，python check_env.py"""
import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys


def section(title):
    print(f"\n=== {title} ===")


def check_import(name, attr=None):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "ok")
        if attr and not hasattr(mod, attr):
            return False, f"{name} 缺少 {attr}"
        return True, ver
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS 推理环境自检")
    parser.add_argument("--quick", action="store_true", help="仅检查关键依赖，启动前快速验证")
    args = parser.parse_args()

    ok = True
    section("系统信息")
    print(f"  系统: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python: {sys.version.split()[0]} @ {sys.executable}")
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    print(f"  虚拟环境: {'是' if in_venv else '否（建议先 source venv/bin/activate）'}")

    section("关键依赖")
    required = [
        "torch",
        "torchaudio",
        "numpy",
        "gradio",
        "transformers",
        "nltk",
        "jieba",
        "LangSegment",
        "librosa",
        "pandas",
        "openpyxl",
    ]
    versions = {}
    for mod in required:
        good, info = check_import(mod)
        mark = "✓" if good else "✗"
        print(f"  {mark} {mod}: {info}")
        if good:
            versions[mod] = info
        else:
            ok = False

    section("系统工具")
    root = os.path.dirname(os.path.abspath(__file__))
    local_ffmpeg = os.path.join(root, "bin", "ffmpeg")
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin and os.access(local_ffmpeg, os.X_OK):
        ffmpeg_bin = local_ffmpeg
    if ffmpeg_bin:
        try:
            ver = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
            ).stdout.splitlines()[0]
            print(f"  ✓ ffmpeg: {ver} ({ffmpeg_bin})")
        except Exception:
            print(f"  ✓ ffmpeg: {ffmpeg_bin}")
    else:
        if os.path.isfile(local_ffmpeg) and not os.access(local_ffmpeg, os.X_OK):
            print(f"  ✗ ffmpeg: 发现 {local_ffmpeg} 但不可执行，请 chmod +x bin/ffmpeg")
        else:
            print("  ✗ ffmpeg: 未安装（合成音频必需）")
        if platform.system() == "Darwin":
            print("    安装（无需 brew）: bash install_ffmpeg_mac.sh")
            print("    或: brew install ffmpeg")
        elif platform.system() == "Windows":
            print("    安装: https://ffmpeg.org/download.html 并加入 PATH")
        else:
            print("    安装: sudo apt install ffmpeg")
        ok = False

    if args.quick:
        if not ok:
            print("\n环境检查未通过，请运行: bash setup.sh")
            return 1
        print("\n快速检查通过。")
        return 0

    section("版本兼容性")
    py = sys.version_info
    if py < (3, 10):
        print(f"  ⚠ Python {py.major}.{py.minor}：建议 3.10 或 3.11")
    torch_ver = versions.get("torch", "")
    if torch_ver:
        parts = torch_ver.split("+")[0].split(".")
        if len(parts) >= 2 and (int(parts[0]), int(parts[1])) >= (2, 6):
            print(f"  ⚠ torch {torch_ver}：需使用项目内置 safe_torch_load（已修复）")
        else:
            print(f"  ✓ torch {torch_ver}")

    np_ver = versions.get("numpy", "")
    if np_ver.startswith("2."):
        print(f"  ✗ numpy {np_ver}：需 numpy<2，请重新运行 setup.sh")
        ok = False
    elif np_ver:
        print(f"  ✓ numpy {np_ver}")

    section("项目模块导入")
    root = os.path.dirname(os.path.abspath(__file__))
    gpt = os.path.join(root, "GPT_SoVITS")
    if gpt not in sys.path:
        sys.path.insert(0, gpt)
    sys.path.insert(0, root)

    modules = [
        ("utils", "safe_torch_load"),
        ("feature_extractor", "cnhubert"),
        ("AR.modules.patched_mha_with_cache", "multi_head_attention_forward_patched"),
    ]
    for mod, attr in modules:
        try:
            m = importlib.import_module(mod)
            getattr(m, attr)
            print(f"  ✓ {mod}.{attr}")
        except Exception as e:
            print(f"  ✗ {mod}.{attr}: {e}")
            ok = False

    section("结论")
    if ok:
        print("  环境就绪，可运行: ./start_inference.sh")
        return 0
    print("  环境未就绪，请执行:")
    print("    rm -rf venv && bash setup.sh")
    print("  若仍失败，把本脚本完整输出发给维护者。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
