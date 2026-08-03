#!/usr/bin/env python3
"""安装 LangSegment（PyPI 0.2.0 有 bug，优先用 0.3.5 备份源）。"""
import os
import site
import subprocess
import sys

ZIP_URL = "https://github.com/chameleon-ai/LangSegment-0.3.5-backup/archive/refs/heads/main.zip"
GIT_URL = "LangSegment @ git+https://github.com/chameleon-ai/LangSegment-0.3.5-backup.git"


def pip(*args):
    return subprocess.run([sys.executable, "-m", "pip", *args], check=False)


def patch_pypi_init():
    for sp in site.getsitepackages():
        init = os.path.join(sp, "LangSegment", "__init__.py")
        if not os.path.isfile(init):
            continue
        text = open(init, encoding="utf-8").read()
        if "setLangfilters" not in text:
            return True
        new = text.replace("setLangfilters,getLangfilters,", "").replace(
            "setLangfilters, getLangfilters, ", ""
        )
        if new != text:
            open(init, "w", encoding="utf-8").write(new)
            print(f"已修补 LangSegment: {init}")
        return True
    return False


def verify():
    import LangSegment  # noqa: F401

    print(f"LangSegment 安装成功: {getattr(LangSegment, '__version__', 'unknown')}")
    return True


def main():
    pip("uninstall", "-y", "LangSegment")

    methods = [
        ("GitHub zip（无需 git）", ["install", ZIP_URL]),
        ("GitHub git", ["install", GIT_URL]),
        ("PyPI 0.2.0 + 自动修补", ["install", "LangSegment==0.2.0"]),
    ]

    for name, args in methods:
        print(f"尝试通过 {name} 安装 LangSegment ...")
        if pip(*args).returncode != 0:
            continue
        if "0.2.0" in name and not patch_pypi_init():
            continue
        try:
            verify()
            return 0
        except Exception as e:
            print(f"安装后验证失败: {e}")
            pip("uninstall", "-y", "LangSegment")

    print("LangSegment 全部安装方式均失败，请检查网络后重试。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
