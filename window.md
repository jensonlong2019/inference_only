# Windows 使用说明

本文档用于配合 Windows 分发包使用。你只需要把压缩包拷到 Windows 机器，解压后按步骤执行即可。

## 1. 环境准备

- 操作系统：Windows 10/11（64 位）
- Python：建议 `3.10` 或 `3.11`（安装时勾选 `Add python.exe to PATH`）
- FFmpeg：必须安装并加入 `PATH`

可在命令行先自检：

```bat
python --version
ffmpeg -version
```

若两条命令都能输出版本号，说明基础环境 OK。

## 2. 解压与目录确认

解压后目录应类似：

```text
GPT-SoVITS-Inference-Windows/
├── GPT_SoVITS/
├── tools/
├── output/
│   └── batch_result/
├── start_inference.bat
├── setup_windows.bat
├── requirements.txt
└── ...
```

说明：`output/batch_result` 已经预创建，便于直接产出音频文件。

## 3. 首次安装依赖（只做一次）

优先用 `CMD` 运行（不要用 PowerShell 直接双击未知脚本）。  
在解压目录打开 `CMD` 后执行：

```bat
cd /d 你的解压目录
setup_windows.bat
```

如果执行后看起来“没反应”，请先看同目录日志文件：

```bat
type setup_windows.log
```

它会自动执行：

- 检查 Python
- 创建 `venv` 虚拟环境
- 升级 `pip`
- 安装 CPU 版 `torch` / `torchaudio`
- 安装 `requirements.txt` 依赖
- 自动启动 `start_inference.bat`

说明：首次安装耗时较长；后续再次运行 `setup_windows.bat` 会自动跳过已健康安装的 VC++ Runtime 和 `torch/torchaudio`，速度会明显更快。
首次安装可能下载大体积依赖（如 `mkl` 约 228MB），弱网环境下会出现超时后自动续传，属于正常现象。

## 4. 放置模型权重

请把你的模型文件放到以下目录：

- GPT 模型（`.ckpt`）放到 `GPT_weights_v2/`
- SoVITS 模型（`.pth`）放到 `SoVITS_weights_v2/`

如果你用的是旧版目录，也可放到：

- `GPT_weights/`
- `SoVITS_weights/`

## 5. 启动推理服务

在同一个目录中执行：

```bat
start_inference.bat
```

启动日志会写入同目录 `start_inference.log`，便于排查错误。

默认端口是 `9872`，启动后浏览器会打开 WebUI（或手动访问 `http://127.0.0.1:9872`）。

## 6. 可选：修改端口/语言

在 CMD 中运行：

```bat
set infer_ttswebui=9874
set language=zh_CN
start_inference.bat
```

## 7. 常见问题

### 7.1 `python` 命令不存在

- 重新安装 Python 3.10/3.11
- 勾选 `Add python.exe to PATH`
- 重新打开 CMD 再试

### 7.2 `ffmpeg` 命令不存在

- 安装 FFmpeg
- 将其 `bin` 目录加入系统 `PATH`
- 重新打开 CMD 验证 `ffmpeg -version`

### 7.3 依赖安装失败

- 检查网络连接
- 关闭代理后重试
- 删除 `venv` 后重新运行 `setup_windows.bat`
- 可查看日志：`setup_windows.log`
- 如果日志包含 `jieba_fast` / `Microsoft Visual C++ 14.0`，可忽略该包并继续（当前版本已默认使用 `jieba` 作为回退）
- 手动兜底安装（CPU 版）：

```bat
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### 7.6 `c10.dll` / `WinError 1114` / DLL 初始化失败

- 这是 `torch` 依赖的运行库问题，最新 `setup_windows.bat` 已自动安装 VC++ Runtime 并重装 torch
- 请先完整重新执行一次 `setup_windows.bat`
- 如果仍报错，重启一次 Windows 后再执行 `setup_windows.bat`
- 仍失败时，手动安装 VC++ Runtime（x64）：<https://aka.ms/vs/17/release/vc_redist.x64.exe>
- 安装后再次执行 `setup_windows.bat`，并把 `setup_windows.log` 发给维护者定位

### 7.7 `ImportError: cannot import name 'HfFolder'`

- 这是 `gradio` 与 `huggingface_hub` 版本不兼容导致
- 最新 `setup_windows.bat` 会自动固定兼容版本
- 如历史环境已污染，删除 `venv` 后重新执行 `setup_windows.bat`

### 7.8 `ModuleNotFoundError: No module named 'wordsegment'`

- 这是英文前处理依赖缺失
- 最新依赖已包含 `wordsegment` 和 `g2p_en`
- 重新执行 `setup_windows.bat` 即可自动补齐

### 7.9 `Failed to initialize NumPy: _ARRAY_API not found`

- 这是 `torch` 与 `numpy 2.x` 的兼容问题
- 最新安装脚本会自动固定为 `numpy<2`
- 如历史环境已安装 `numpy 2.x`，重新执行 `setup_windows.bat` 会自动降级

### 7.10 `ValueError` 提示 `torch.load` 需要 `torch>=2.6`

- 这是 `transformers` 版本过新与当前 `torch` 版本不兼容
- 最新安装脚本会自动固定 `transformers<4.46.0`
- 删除旧 `venv` 后重新执行 `setup_windows.bat` 可彻底生效

### 7.11 `ValueError: When localhost is not accessible...`

- 这是本机代理/网络策略影响 `gradio` 的 localhost 检测
- 最新版本已在 Windows 默认使用 `127.0.0.1` 启动，并设置 `NO_PROXY=127.0.0.1,localhost`
- 同时会在启动脚本中清空 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`，避免代理劫持本地回环
- 若仍触发该错误，程序会自动回退到 `share=True` 模式重试启动
- 启动后请手动在浏览器打开：`http://127.0.0.1:9872`

### 7.12 `TypeError: argument of type 'bool' is not iterable`（gradio_client）

- 这是部分 `gradio/gradio_client` 组合在解析 schema 时的兼容问题
- 最新版本已内置启动时兼容补丁（同时覆盖 `TypeError` 和 `APIInfoParseError: Cannot parse schema True`）
- 如果仍出现，删除旧 `venv` 后重新执行 `setup_windows.bat`

### 7.13 `TypeError: unhashable type: 'dict'`（gradio/starlette）

- 这是 `gradio` 与 `fastapi/starlette` 版本矩阵不匹配导致
- 最新安装脚本已固定：`fastapi==0.112.4`、`starlette==0.38.2`
- **当前版本已在代码中内置自动猴子补丁（Monkeypatch），通常不再需要手动干预。**
- 若仍出现，请删除旧 `venv` 后重新执行 `setup_windows.bat`。

### 7.4 端口被占用

- 改端口启动（见第 6 节）
- 或结束占用该端口的进程

### 7.5 打开 `bat` 后一堆“不是内部或外部命令”

- 原因通常是脚本编码和系统命令行编码不兼容
- 请使用本包内最新的 `setup_windows.bat` / `start_inference.bat`
- 从 `CMD` 进入目录后手动执行，不要用其他编辑器改编码后再运行

---

如果要重新分发给其他人，建议先在源目录运行 `clean_for_sharing.bat` 清理临时文件，再重新打包。
