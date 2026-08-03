# GPT-SoVITS 推理功能独立版

这是从 GPT-SoVITS 项目中提取的**纯推理功能**模块，不包含训练、数据准备等其他功能。

## 📁 目录结构

```
inference_only/
├── GPT_SoVITS/          # 核心推理代码
│   ├── AR/              # 自回归模型
│   ├── module/          # 模型模块
│   ├── feature_extractor/  # 特征提取器
│   ├── text/            # 文本处理
│   ├── configs/         # 配置文件
│   ├── pretrained_models/  # 预训练模型
│   ├── inference_webui.py  # WebUI 主文件
│   └── utils.py
├── tools/               # 工具函数
│   ├── my_utils.py      # 工具函数
│   └── i18n/            # 国际化
├── start_inference.sh   # Linux/Mac 启动脚本
├── start_inference.bat  # Windows 启动脚本
└── README.md           # 本文件
```

## 🚀 快速开始

### 方式一：使用项目虚拟环境（推荐）

如果 `inference_only` 在项目目录 `GPT-SoVITS-v2-2025` 下，启动脚本会自动检测并使用项目的虚拟环境。

### 方式二：创建独立虚拟环境

如果 `inference_only` 在独立位置，需要先创建虚拟环境：

```bash
# Linux/Mac
cd inference_only
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod +x start_inference.sh
./start_inference.sh

# Windows
cd inference_only
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
start_inference.bat
```

### 方式三：使用系统 Python（不推荐）

如果系统 Python 已安装所有依赖，可以直接运行：

```bash
# Linux/Mac
cd inference_only
./start_inference.sh

# Windows
cd inference_only
start_inference.bat
```

**注意**：启动脚本会自动检测虚拟环境，按以下优先级：
1. 当前目录下的 `venv/`
2. 项目目录 `GPT-SoVITS-v2-2025/venv/`
3. 上级目录的 `venv/`
4. 系统 Python（如果已安装依赖）

### 自定义端口和语言

```bash
# Linux/Mac
export infer_ttswebui=9874
export language=zh_CN
./start_inference.sh

# Windows
set infer_ttswebui=9874
set language=zh_CN
start_inference.bat
```

## 📋 功能特性

- ✅ **单条生成**：输入文本直接生成语音
- ✅ **批量生成**：支持 Excel 批量处理
- ✅ **多语种支持**：中文、英文、日文、韩文等
- ✅ **模型切换**：支持切换不同的 GPT 和 SoVITS 模型
- ✅ **音量调节**：支持全局音量控制
- ✅ **格式输出**：支持 WAV 和 MP3 格式

## 🔧 依赖要求

确保已安装以下 Python 包：

```bash
pip install torch torchaudio
pip install gradio
pip install transformers
pip install librosa
pip install pandas openpyxl
pip install scipy numpy
pip install LangSegment
```

## 📝 使用说明

1. **启动服务**：运行启动脚本后，会在浏览器中自动打开 WebUI 界面
2. **上传参考音频**：选择 3-10 秒的参考音频
3. **输入文本**：在文本框中输入要合成的文本
4. **选择语种**：根据文本内容选择对应的语种
5. **生成语音**：点击"合成语音"按钮即可生成

## 🎯 批量生成

1. 准备 Excel 文件，包含"文件名"和"配音内容"两列
2. 上传 Excel 文件
3. 选择对应的列
4. 点击"开始批量生成"
5. 结果会保存在 `output/batch_result` 目录

## ⚙️ 配置说明

- **端口**：默认 9872，可通过环境变量 `infer_ttswebui` 修改
- **语言**：默认 Auto，可通过环境变量 `language` 修改
- **模型路径**：模型文件应放在 `GPT_weights_v2/` 和 `SoVITS_weights_v2/` 目录

## 📦 添加自己的模型权重

如果您有自己的训练好的模型，请按以下步骤添加：

1. **GPT 模型**：将 `.ckpt` 文件放入 `GPT_weights_v2/` 目录
2. **SoVITS 模型**：将 `.pth` 文件放入 `SoVITS_weights_v2/` 目录
3. 启动 WebUI 后，在界面上选择对应的模型即可使用

## 📤 分享给他人

### 准备分享包

在分享前，建议先清理不必要的文件：

**Linux/Mac:**
```bash
chmod +x clean_for_sharing.sh
./clean_for_sharing.sh
```

**Windows:**
```cmd
clean_for_sharing.bat
```

### 压缩打包

```bash
# Linux/Mac
zip -r GPT-SoVITS-Inference.zip inference_only/

# Windows
# 右键点击 inference_only 文件夹，选择"发送到" -> "压缩(zipped)文件夹"
```

### 分享内容

压缩包包含：
- ✅ 所有核心推理代码
- ✅ 预训练模型（基础模型）
- ✅ 启动脚本和依赖列表
- ✅ 使用文档

**不包含**（需要接收者自己添加）：
- ❌ 用户训练的模型权重（需要放在 `GPT_weights_v2/` 和 `SoVITS_weights_v2/`）
- ❌ Python 虚拟环境（接收者需要自己安装依赖）
- ❌ 用户生成的文件和缓存

## 📌 注意事项

1. 首次运行可能需要下载一些 NLTK 资源，请保持网络连接
2. 如果使用 CUDA，确保已正确安装 CUDA 和 PyTorch GPU 版本
3. 批量生成时建议使用"按标点符号切"模式，避免吃字问题
4. 对于包含英文字母的中文名词（如"验电器A"），可以使用 `【】` 括起来确保连贯性

## 🐛 常见问题

**Q: 启动后提示找不到模块？**
A: 确保在 `inference_only` 目录下运行启动脚本，脚本会自动设置 Python 路径。

**Q: 端口被占用？**
A: 修改环境变量 `infer_ttswebui` 指定其他端口。

**Q: 批量生成结果预览为空？**
A: 确保生成完成后等待几秒，或刷新页面。

## 📄 许可证

与原项目保持一致。

# inference_only
# inference_only
