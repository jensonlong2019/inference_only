#!/usr/bin/env python3
"""环境自检：安装后或排错时运行，python3 check_env.py"""
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

    deps_ok = True
    models_ok = True

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
            deps_ok = False

    np_ver = versions.get("numpy", "")
    if np_ver.startswith("2."):
        print(f"  ✗ numpy {np_ver}：需 numpy<2，请执行: pip install 'numpy>=1.26,<2'")
        deps_ok = False

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
        deps_ok = False

    section("模型权重（Git 不含大文件，需单独拷贝）")
    pretrained = os.path.join(root, "GPT_SoVITS", "pretrained_models")
    pretrained_required = [
        "chinese-hubert-base/pytorch_model.bin",
        "chinese-roberta-wwm-ext-large/pytorch_model.bin",
    ]
    if os.path.isdir(pretrained):
        missing_pre = [
            p
            for p in pretrained_required
            if not os.path.isfile(os.path.join(pretrained, p))
        ]
        if missing_pre:
            print(f"  ✗ pretrained_models 不完整，缺少: {', '.join(missing_pre)}")
            models_ok = False
        else:
            print("  ✓ GPT_SoVITS/pretrained_models")
    else:
        print("  ✗ 缺少 GPT_SoVITS/pretrained_models/（约 1.4G，需单独转发）")
        models_ok = False

    def _has_weight(dirs, exts, also_paths=None):
        for d in dirs:
            dp = os.path.join(root, d)
            if not os.path.isdir(dp):
                continue
            for name in os.listdir(dp):
                if any(name.endswith(e) for e in exts):
                    return True
        for p in also_paths or []:
            if os.path.isfile(os.path.join(root, p)):
                return True
        return False

    gpt_ok = _has_weight(
        ["GPT_weights_v2", "GPT_weights"],
        [".ckpt"],
        also_paths=[
            "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        ],
    )
    sovits_ok = _has_weight(
        ["SoVITS_weights_v2", "SoVITS_weights"],
        [".pth"],
        also_paths=[
            "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
            "GPT_SoVITS/pretrained_models/s2G488k.pth",
        ],
    )
    print(f"  {'✓' if gpt_ok else '✗'} GPT 权重 (.ckpt)")
    print(f"  {'✓' if sovits_ok else '✗'} SoVITS 权重 (.pth)")
    if not gpt_ok or not sovits_ok:
        print("    请拷贝 GPT_weights_v2/、SoVITS_weights_v2/，或完整 pretrained_models/")
        models_ok = False

    g2pw = os.path.join(root, "GPT_SoVITS", "text", "G2PWModel", "g2pW.onnx")
    if os.path.isfile(g2pw):
        print("  ✓ G2PWModel/g2pW.onnx")
    else:
        print("  ⚠ 缺少 GPT_SoVITS/text/G2PWModel/g2pW.onnx（中文多音字可能受影响）")

    ok = deps_ok and models_ok

    if args.quick:
        # setup.sh 只要求依赖通过；权重缺失单独提示，不判失败
        if not deps_ok:
            print("\n依赖检查未通过，请运行: bash setup.sh")
            return 1
        if not models_ok:
            print("\n依赖已就绪。下一步：向维护者索取并拷贝模型目录后再启动。")
            print("  - GPT_SoVITS/pretrained_models/")
            print("  - GPT_weights_v2/ 与 SoVITS_weights_v2/")
            print("  - 建议: GPT_SoVITS/text/G2PWModel/")
            return 0
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
    if np_ver and not np_ver.startswith("2."):
        print(f"  ✓ numpy {np_ver}")

    section("项目模块导入")
    gpt = os.path.join(root, "GPT_SoVITS")
    if gpt not in sys.path:
        sys.path.insert(0, gpt)
    if root not in sys.path:
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
            deps_ok = False

    ok = deps_ok and models_ok
    section("结论")
    if ok:
        print("  环境就绪，可运行: ./start_inference.sh")
        return 0
    if not deps_ok:
        print("  依赖未就绪，请执行:")
        print("    rm -rf venv && bash setup.sh")
        print("  注意 macOS/Linux 用 python3，不要用 python")
    if not models_ok:
        print("  权重未就绪，请向维护者索取并拷贝:")
        print("    GPT_SoVITS/pretrained_models/")
        print("    GPT_weights_v2/ 与 SoVITS_weights_v2/")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
