import os, sys, pdb, torch

# 确保 GPT_SoVITS 内子模块（feature_extractor / module / AR 等）可被导入
_GPT_SOVITS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GPT_SOVITS_DIR)
for _p in (_GPT_SOVITS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 【紧急修复】解决 Gradio/FastAPI 兼容性导致的 "TypeError: unhashable type: 'dict'"
# 必须在导入 gradio 之前执行
try:
    import jinja2
    from starlette.templating import Jinja2Templates
    
    _original_get_template = Jinja2Templates.get_template
    def _patched_get_template(self, name):
        # 如果 name 是 dict（通常是由于版本兼容性导致参数错位），尝试修复
        if isinstance(name, dict):
            return _original_get_template(self, "")
        return _original_get_template(self, name)
    Jinja2Templates.get_template = _patched_get_template
except Exception:
    pass

import logging
import traceback
import subprocess
import ssl
import nltk

# 【自动修复】检查并安装 openpyxl (用于读取 Excel)
try:
    import openpyxl
except ImportError:
    print("检测到缺失 openpyxl，正在尝试自动安装...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
        print("openpyxl 安装成功！")
    except Exception as e:
        print(f"自动安装 openpyxl 失败: {e}。建议手动安装或使用 CSV 文件。")

# 【自动修复】检查并下载必要的 NLTK 资源
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk_packages = ['averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'cmudict']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'taggers/{pkg}' if 'tagger' in pkg else f'corpora/{pkg}')
    except LookupError:
        print(f"正在下载缺失的 NLTK 资源: {pkg}...")
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)

# 兼容性处理：LangSegment 不同版本的导入方式
try:
    import LangSegment
    # 尝试检查是否有 setfilters 方法
    if not hasattr(LangSegment, 'setfilters'):
        # 某些版本可能需要从子模块导入
        try:
            from LangSegment import LangSegment as LangSegmentCore
            LangSegment.setfilters = LangSegmentCore.setfilters if hasattr(LangSegmentCore, 'setfilters') else lambda x: None
            LangSegment.getTexts = LangSegmentCore.getTexts if hasattr(LangSegmentCore, 'getTexts') else lambda x: [{"text": x, "lang": "zh"}]
        except:
            # 如果都失败，创建一个兼容的包装
            def setfilters(filters):
                pass  # 某些版本可能不需要设置
            def getTexts(text):
                return [{"text": text, "lang": "zh"}]
            LangSegment.setfilters = setfilters
            LangSegment.getTexts = getTexts
except ImportError as e:
    print(f"警告: LangSegment 导入失败: {e}")
    # 创建一个最小兼容实现
    class LangSegment:
        @staticmethod
        def setfilters(filters):
            pass
        @staticmethod
        def getTexts(text):
            return [{"text": text, "lang": "zh"}]
    print("已使用兼容模式，语种自动识别功能可能受限")

import re, json
import pandas as pd # 导入pandas
from time import time as ttime

version=os.environ.get("version","v2")
pretrained_sovits_name=["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", "GPT_SoVITS/pretrained_models/s2G488k.pth"]
pretrained_gpt_name=["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"]

_ =[[],[]]
for i in range(2):
    if os.path.exists(pretrained_gpt_name[i]):
        _[0].append(pretrained_gpt_name[i])
    if os.path.exists(pretrained_sovits_name[i]):
        _[-1].append(pretrained_sovits_name[i])
pretrained_gpt_name,pretrained_sovits_name = _

if os.path.exists(f"./weight.json"):
    pass
else:
    with open(f"./weight.json", 'w', encoding="utf-8") as file:json.dump({'GPT':{},'SoVITS':{}},file)

def _first_existing_weight(candidates, roots, exts):
    """从候选路径 / 权重目录中挑出第一个存在的模型文件。"""
    if isinstance(candidates, str):
        candidates = [candidates]
    for p in candidates or []:
        if p and os.path.isfile(p):
            return p
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if any(name.endswith(ext) for ext in exts):
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    return path
    return None

with open(f"./weight.json", 'r', encoding="utf-8") as file:
    weight_data = file.read()
    weight_data=json.loads(weight_data)
    gpt_path = os.environ.get(
        "gpt_path", weight_data.get('GPT',{}).get(version,pretrained_gpt_name))
    sovits_path = os.environ.get(
        "sovits_path", weight_data.get('SoVITS',{}).get(version,pretrained_sovits_name))
    # weight.json / 环境变量优先；不存在则回退到 pretrained 与权重目录扫描
    def _merge_candidates(primary, fallbacks):
        out = []
        for item in ([primary] if isinstance(primary, str) else (primary or [])):
            if item and item not in out:
                out.append(item)
        for item in fallbacks or []:
            if item and item not in out:
                out.append(item)
        return out
    gpt_path = _first_existing_weight(
        _merge_candidates(gpt_path, pretrained_gpt_name),
        ["GPT_weights_v2", "GPT_weights"],
        [".ckpt"],
    )
    sovits_path = _first_existing_weight(
        _merge_candidates(sovits_path, pretrained_sovits_name),
        ["SoVITS_weights_v2", "SoVITS_weights"],
        [".pth"],
    )
    if not gpt_path or not sovits_path:
        print("=" * 60)
        print("错误: 未找到可用的 GPT / SoVITS 模型权重，无法启动。")
        print("")
        print("请把以下目录完整拷贝到本项目根目录（与 GPT_SoVITS 同级）：")
        print("  1) GPT_SoVITS/pretrained_models/   # 预训练底座（必需）")
        print("  2) GPT_weights_v2/                 # 微调 GPT .ckpt（可选，有则优先）")
        print("  3) SoVITS_weights_v2/              # 微调 SoVITS .pth（可选，有则优先）")
        print("")
        print(f"当前 GPT:    {gpt_path or '未找到'}")
        print(f"当前 SoVITS: {sovits_path or '未找到'}")
        print("=" * 60)
        sys.exit(1)

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
punctuation = set(['!', '?', '…', ',', '.', '-'," "])
import gradio as gr
try:
    import gradio_client.utils as gradio_client_utils
    _orig_get_type = gradio_client_utils.get_type
    _orig_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type

    def _safe_get_type(schema):
        # Some gradio_client versions crash when schema is bool.
        if isinstance(schema, bool):
            return "Any"
        return _orig_get_type(schema)

    def _safe_json_schema_to_python_type(schema, defs=None):
        # Handle JSON schema boolean shorthand (True/False) used by some pydantic outputs.
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return _orig_json_schema_to_python_type(schema, defs)

    gradio_client_utils.get_type = _safe_get_type
    gradio_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
except Exception:
    pass
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from utils import safe_torch_load
from text import cleaned_text_to_sequence
from text import english
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio
from tools.i18n.i18n import I18nAuto, scan_language_list

language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dict_language_v1 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("粤语"): "all_yue",#全部按中文识别
    i18n("韩文"): "all_ko",#全部按韩文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("粤英混合"): "yue",#按粤英混合识别####不变
    i18n("韩英混合"): "ko",#按韩英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",#多语种启动切分识别语种
}
dict_language = dict_language_v1 if version =='v1' else dict_language_v2

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def change_sovits_weights(sovits_path,prompt_language=None,text_language=None):
    global vq_model, hps, version, dict_language
    dict_s2 = safe_torch_load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    # print("sovits版本:",hps.model.version)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    dict_language = dict_language_v1 if version =='v1' else dict_language_v2
    with open("./weight.json")as f:
        data=f.read()
        data=json.loads(data)
        data["SoVITS"][version]=sovits_path
    with open("./weight.json","w")as f:f.write(json.dumps(data))
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = {'__type__':'update'},  {'__type__':'update', 'value':prompt_language}
        else:
            prompt_text_update = {'__type__':'update', 'value':''}
            prompt_language_update = {'__type__':'update', 'value':i18n("中文")}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {'__type__':'update'}, {'__type__':'update', 'value':text_language}
        else:
            text_update = {'__type__':'update', 'value':''}
            text_language_update = {'__type__':'update', 'value':i18n("中文")}
        return  {'__type__':'update', 'choices':list(dict_language.keys())}, {'__type__':'update', 'choices':list(dict_language.keys())}, prompt_text_update, prompt_language_update, text_update, text_language_update



change_sovits_weights(sovits_path)


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = safe_torch_load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./weight.json")as f:
        data=f.read()
        data=json.loads(data)
        data["GPT"][version]=gpt_path
    with open("./weight.json","w")as f:f.write(json.dumps(data))


change_gpt_weights(gpt_path)


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx=audio.abs().max()
    if(maxx>1):audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

# ABCD/S 字母音频文件夹路径
ABCD_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "ABCD")

def extract_letter_prefix(text):
    """
    检测文本是否以单个字母开头（如 "A."、"B."、"S." 等）
    返回: (letter, remaining_text) 或 (None, text)
    例如: "A.蜡笔" -> ("A", "蜡笔"), "A." -> ("A", ""), "S" -> ("S", "")
    """
    text = text.strip()
    # 匹配单个大写字母后跟标点（.、。、,、，等），然后跟后续文本
    # 例如: "A.蜡笔" -> letter="A", remaining="蜡笔"
    #      "A." -> letter="A", remaining=""
    match = re.match(r'^([A-Z])([.。，,、])(.*)$', text)
    if match:
        letter = match.group(1)
        remaining = match.group(3).strip()
        return letter, remaining
    # 如果没有标点，但单个字母后直接跟中文或其他非字母字符，也认为是字母前缀
    # 例如: "A蜡笔" -> letter="A", remaining="蜡笔"
    match = re.match(r'^([A-Z])([^A-Za-z].*)$', text)
    if match:
        letter = match.group(1)
        remaining = match.group(2).strip()
        return letter, remaining
    # 如果整行就是一个字母（没有后续内容）
    # 例如: "S" -> letter="S", remaining=""
    match = re.match(r'^([A-Z])$', text)
    if match:
        letter = match.group(1)
        return letter, ""
    return None, text

def load_letter_audio(letter, sample_rate):
    """
    加载字母音频文件
    返回: numpy 数组的音频数据，如果文件不存在则返回 None
    """
    audio_path = os.path.join(ABCD_AUDIO_DIR, f"{letter}.wav")
    if os.path.exists(audio_path):
        try:
            audio_data = load_audio(audio_path, sample_rate)
            return audio_data
        except Exception as e:
            print(f"加载字母音频 {audio_path} 失败: {e}")
            return None
    return None

_caps_abbrev_re = re.compile(r'(?<![A-Za-z])([A-Z]{2,6})(?![A-Za-z])')

def split_caps_abbrev(text):
    """
    将全大写短词拆成逐字母（如 ABC -> A B C），避免被当作语气拖长
    仅处理 2~6 位的全大写词，且两侧不接字母
    """
    def repl(match):
        word = match.group(1)
        return " ".join(list(word))
    return _caps_abbrev_re.sub(repl, text)

from text import chinese
def get_phones_and_bert(text,language,version,final=False):
    # 彻底解决“语种切分”导致的名词中断问题
    # 方案：在任何语种判断之前，先提取受保护的【】内容，并强制其为连贯片段
    if "【" in text and "】" in text:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        
        # 1. 手动按括号切分
        parts = re.split(r'(【.*?】)', text)
        for part in parts:
            if not part: continue
            if part.startswith("【") and part.endswith("】"):
                # 2. 受保护区域：去掉括号，强制标记为 zh
                # GPT-SoVITS 的中文推理模式完美支持中英混排，且不会像 LangSegment 那样强行切断
                content = part[1:-1]
                textlist.append(content)
                langlist.append("zh")
            else:
                # 3. 非保护区域：根据当前模式进行切分
                if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
                    # 这些是“全语种”模式，不需要 LangSegment
                    textlist.append(part)
                    langlist.append(language.replace("all_", ""))
                else:
                    # 这些是“混合”模式，需要 LangSegment
                    for tmp in LangSegment.getTexts(part):
                        l = tmp["lang"]
                        if language == "auto_yue" and l == "zh": l = "yue"
                        elif language not in ["auto", "auto_yue"] and l != "en": l = language
                        langlist.append(l)
                        textlist.append(tmp["text"])
        
        # 直接跳到特征提取环节，不再走下方的递归和切分
        print(f"DEBUG: Bracket mode - segments: {textlist}, langs: {langlist}")
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            content = textlist[i]
            
            # 核心修复：如果受保护片段包含英文字母，使用 mix_text_normalize 确保字母被正确发音
            if lang == "zh" and re.search(r'[A-Za-z0-9]', content):
                # 将小写字母转为大写，然后使用 mix_text_normalize 处理中英混排
                content = re.sub(r'[a-z]', lambda x: x.group(0).upper(), content)
                content = split_caps_abbrev(content)
                content = chinese.mix_text_normalize(content)
                # 递归调用 get_phones_and_bert，但传入 final=True 和已处理的文本（不包含括号）
                # 这样会走 "zh" 模式下的中英混排分支，但不会再次触发括号检测
                phones, bert_seg, norm_text = get_phones_and_bert(content, "zh", version, final=True)
                # bert_seg 已经是处理好的 bert 特征，直接使用
                bert_list.append(bert_seg)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
            else:
                phones, word2ph, norm_text = clean_text_inf(content, lang, version)
                bert = get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)
        
        if not final and len(phones) < 6:
            return get_phones_and_bert("." + text,language,version,final=True)
        return phones,bert.to(dtype),norm_text

    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 针对全语种模式，直接去掉保护括号，内部内容本身就不会被切分
            formattext = text.replace("【", "").replace("】", "")
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "zh":
            if re.search(r'[A-Za-z0-9]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = split_caps_abbrev(formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        # 【优化方案】对于短文本的中英混合，不切分，手动识别中英文部分，分别处理音素后合并
        # 这样音素是连贯的，BERT特征也是连贯的（英文部分用全零，不会产生突变）
        if language == "zh" and len(text) <= 20 and re.search(r'[A-Za-z]', text) and re.search(r'[\u4e00-\u9fa5]', text):
            # 手动识别中英文部分（不切分）
            segments = []
            current_seg = ""
            current_lang = None
            
            for char in text:
                is_chinese = '\u4e00' <= char <= '\u9fff'
                is_english = char.isalpha()
                
                if is_chinese:
                    if current_lang == "en":
                        segments.append(("en", current_seg))
                        current_seg = char
                        current_lang = "zh"
                    else:
                        current_seg += char
                        current_lang = "zh"
                elif is_english:
                    if current_lang == "zh":
                        segments.append(("zh", current_seg))
                        current_seg = char
                        current_lang = "en"
                    else:
                        current_seg += char
                        current_lang = "en"
                else:
                    # 标点符号等，跟随当前语言
                    current_seg += char
            
            if current_seg:
                segments.append((current_lang, current_seg))
            
            # 如果确实有中英文混合，使用特殊处理
            if len(segments) > 1:
                # 按顺序处理每个segment，先收集所有信息
                segment_info = []
                for seg_lang, seg_text in segments:
                    if seg_lang == "zh":
                        phones, word2ph, norm_text = clean_text_inf(seg_text, "zh", version)
                        segment_info.append(("zh", phones, word2ph, norm_text, seg_text))
                    elif seg_lang == "en":
                        phones_en, _, norm_text_en = clean_text_inf(seg_text, "en", version)
                        segment_info.append(("en", phones_en, None, norm_text_en, seg_text))
                
                # 第一步：先处理所有中文片段，生成所有BERT特征
                phones_list = []
                bert_segments = []
                norm_text_list = []
                zh_bert_dict = {}  # 存储每个中文片段的BERT特征
                
                # 第一遍：只处理中文片段，生成所有BERT特征
                for i, (seg_lang, phones, word2ph, norm_text, seg_text) in enumerate(segment_info):
                    if seg_lang == "zh":
                        # 生成中文BERT特征
                        bert_seg = get_bert_feature(norm_text, word2ph).to(device)
                        zh_bert_dict[i] = bert_seg
                
                # 第二遍：按顺序处理所有片段，生成音素和BERT特征
                for i, (seg_lang, phones, word2ph, norm_text, seg_text) in enumerate(segment_info):
                    if seg_lang == "zh":
                        phones_list.append(phones)
                        norm_text_list.append(norm_text)
                        # 使用已生成的中文BERT特征
                        bert_segments.append(zh_bert_dict[i])
                    elif seg_lang == "en":
                        phones_list.append(phones)
                        norm_text_list.append(norm_text)
                        en_bert_len = len(phones)
                        
                        # 查找前后中文片段的BERT特征（此时所有中文BERT都已生成）
                        prev_zh_bert = None
                        next_zh_bert = None
                        
                        # 向前查找中文片段
                        for j in range(i - 1, -1, -1):
                            if segment_info[j][0] == "zh":
                                prev_zh_bert = zh_bert_dict[j]
                                break
                        
                        # 向后查找中文片段
                        for j in range(i + 1, len(segment_info)):
                            if segment_info[j][0] == "zh":
                                next_zh_bert = zh_bert_dict[j]
                                break
                        
                        # 生成英文部分的BERT特征
                        if prev_zh_bert is not None and next_zh_bert is not None:
                            # 前后都有中文：使用线性插值
                            prev_tail = prev_zh_bert[:, -1:]  # 前一个中文的最后1帧
                            next_head = next_zh_bert[:, :1]   # 后一个中文的第一帧
                            # 线性插值：从prev_tail到next_head
                            bert_en = torch.zeros(
                                (1024, en_bert_len),
                                dtype=torch.float16 if is_half == True else torch.float32,
                                device=device
                            )
                            for k in range(en_bert_len):
                                alpha = (k + 1) / (en_bert_len + 1)
                                bert_en[:, k] = (1 - alpha) * prev_tail.squeeze() + alpha * next_head.squeeze()
                        elif prev_zh_bert is not None:
                            # 只有前面的中文：使用前一个中文末尾特征，平滑衰减
                            tail_frames = min(5, prev_zh_bert.shape[1])
                            bert_tail = prev_zh_bert[:, -tail_frames:]
                            # 使用最后1帧作为基础，然后平滑衰减
                            base_frame = prev_zh_bert[:, -1:]
                            if en_bert_len == 1:
                                bert_en = base_frame
                            else:
                                # 创建衰减系数：从1.0衰减到0.7
                                decay = torch.linspace(1.0, 0.7, en_bert_len, device=device, dtype=base_frame.dtype)
                                bert_en = base_frame.repeat(1, en_bert_len) * decay.unsqueeze(0)
                        elif next_zh_bert is not None:
                            # 只有后面的中文：使用后一个中文开头特征，平滑增强
                            head_frames = min(5, next_zh_bert.shape[1])
                            bert_head = next_zh_bert[:, :head_frames]
                            # 使用第一帧作为基础，然后平滑增强
                            base_frame = next_zh_bert[:, :1]
                            if en_bert_len == 1:
                                bert_en = base_frame
                            else:
                                # 创建增强系数：从0.7增强到1.0
                                enhance = torch.linspace(0.7, 1.0, en_bert_len, device=device, dtype=base_frame.dtype)
                                bert_en = base_frame.repeat(1, en_bert_len) * enhance.unsqueeze(0)
                        else:
                            # 没有中文片段：使用全零（这种情况很少见）
                            bert_en = torch.zeros(
                                (1024, en_bert_len),
                                dtype=torch.float16 if is_half == True else torch.float32,
                                device=device
                            )
                        
                        bert_segments.append(bert_en)
                
                # 按顺序拼接BERT特征和音素
                bert = torch.cat(bert_segments, dim=1)
                phones = sum(phones_list, [])
                norm_text = ''.join(norm_text_list)
            else:
                # 没有中英文混合，走正常流程
                textlist=[]
                langlist=[]
                LangSegment.setfilters(["zh","ja","en","ko"])
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        langlist.append(language)
                    textlist.append(tmp["text"])
                
                phones_list = []
                bert_list = []
                norm_text_list = []
                for i in range(len(textlist)):
                    lang = langlist[i]
                    phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
                    bert = get_bert_inf(phones, word2ph, norm_text, lang)
                    phones_list.append(phones)
                    norm_text_list.append(norm_text)
                    bert_list.append(bert)
                bert = torch.cat(bert_list, dim=1)
                phones = sum(phones_list, [])
                norm_text = ''.join(norm_text_list)
        else:
            # 原有的自动切分逻辑
            textlist=[]
            langlist=[]
            LangSegment.setfilters(["zh","ja","en","ko"])
            
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            
            print(textlist)
            print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
                bert = get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text,language,version,final=True)

    return phones,bert.to(dtype),norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

##ref_wav_path+prompt_text+prompt_language+text(单个)+text_language+top_k+top_p+temperature
# cache_tokens={}#暂未实现清理机制
cache= {}
# 全局缓存变量
_ref_audio_cache = {
    "path": None,
    "prompt_semantic": None,
    "bert1": None,
    "phones1": None,
    "norm_text1": None
}

def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free = False,speed=1,if_freeze=False,inp_refs=123, volume=1.0):
    global cache, _ref_audio_cache
    if ref_wav_path:pass
    else:gr.Warning(i18n('请上传参考音频'))
    if text:pass
    else:gr.Warning(i18n('请填入推理文本'))
    t = []
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        # print(i18n("实际输入的参考文本:"), prompt_text)
    text = text.strip("\n")
    
    # print(i18n("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    
    prompt_semantic = None
    
    if not ref_free:
        # 【性能优化】检查缓存
        # 只有当路径、文本、语言完全一致时才使用缓存
        cache_key = f"{ref_wav_path}|{prompt_text}|{prompt_language}"
        
        if _ref_audio_cache["path"] == cache_key and _ref_audio_cache["prompt_semantic"] is not None:
            # print("Hit reference audio cache!")
            prompt_semantic = _ref_audio_cache["prompt_semantic"]
        else:
            # print("Miss cache, computing reference audio features...")
            with torch.no_grad():
                wav16k, sr = librosa.load(ref_wav_path, sr=16000)
                if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                    gr.Warning(i18n("参考音频在3~10秒范围外，请更换！"))
                    raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
                wav16k = torch.from_numpy(wav16k)
                zero_wav_torch = torch.from_numpy(zero_wav)
                if is_half == True:
                    wav16k = wav16k.half().to(device)
                    zero_wav_torch = zero_wav_torch.half().to(device)
                else:
                    wav16k = wav16k.to(device)
                    zero_wav_torch = zero_wav_torch.to(device)
                wav16k = torch.cat([wav16k, zero_wav_torch])
                ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                    "last_hidden_state"
                ].transpose(
                    1, 2
                )  # .float()
                codes = vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                
            # 更新缓存
            _ref_audio_cache["path"] = cache_key
            _ref_audio_cache["prompt_semantic"] = prompt_semantic
            # 清理旧的文本相关缓存，因为参考音频变了，可能关联的文本特征也得变（虽然 strict 来说 BERT 特征只跟文本有关）
            _ref_audio_cache["bert1"] = None
            
        prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1-t0)

    if (how_to_cut == i18n("凑四句一切")):
        text = cut1(text)
    elif (how_to_cut == i18n("凑50字一切")):
        text = cut2(text)
    elif (how_to_cut == i18n("按中文句号。切")):
        text = cut3(text)
    elif (how_to_cut == i18n("按英文句号.切")):
        text = cut4(text)
    elif (how_to_cut == i18n("按标点符号切")):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    # print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    
    if not ref_free:
        # 【性能优化】缓存参考文本的 BERT 特征
        # 如果参考音频没变，且参考文本没变，就直接复用
        if _ref_audio_cache["path"] == cache_key and _ref_audio_cache["bert1"] is not None:
             phones1, bert1, norm_text1 = _ref_audio_cache["phones1"], _ref_audio_cache["bert1"], _ref_audio_cache["norm_text1"]
        else:
             phones1,bert1,norm_text1=get_phones_and_bert(prompt_text, prompt_language, version)
             _ref_audio_cache["phones1"] = phones1
             _ref_audio_cache["bert1"] = bert1
             _ref_audio_cache["norm_text1"] = norm_text1

    for i_text,text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        
        # 【字母音频拼接功能】检测是否有字母前缀（如 "A."、"B." 等）
        letter, remaining_text = extract_letter_prefix(text)
        letter_audio = None
        if letter:
            # 尝试加载字母音频
            letter_audio = load_letter_audio(letter, hps.data.sampling_rate)
            if letter_audio is not None:
                # 如果成功加载字母音频，使用后续文本进行生成
                if remaining_text:
                    # 有后续文本，使用后续文本
                    text = remaining_text
                    print(f"检测到字母前缀 {letter}，将使用字母音频 + 文本音频拼接")
                else:
                    # 没有后续文本（只有字母前缀），只使用字母音频
                    print(f"检测到字母前缀 {letter}，仅使用字母音频")
                    audio_opt.append(letter_audio)
                    if i_text < len(texts) - 1:
                        shorter_zero_wav = np.zeros(
                            int(hps.data.sampling_rate * 0.1),
                            dtype=np.float16 if is_half == True else np.float32,
                        )
                        audio_opt.append(shorter_zero_wav)
                    continue
        
        # 如果文本为空，跳过生成
        if not text.strip():
            continue
        
        # 【修复单个字符文本问题】对于单个字符的文本（特别是字母音频拼接后的），
        # 不自动添加标点，避免影响发音和产生拉尾音
        text_stripped = text.strip()
        is_single_char = len(text_stripped) == 1
        
        # 优化：只在文本确实需要标点时才添加，避免不必要的标点导致语气词
        # 如果文本已经被切分（不是最后一段），且末尾没有标点，才添加标点
        # 这样可以减少因为自动添加标点导致的不自然停顿和语气词
        # 但对于单个字符文本，跳过自动添加标点
        if text_stripped and text_stripped[-1] not in splits and not is_single_char:
            # 优化：只有在后面还有文字段落时才加标点
            # 这样全文最后一段不会被强制补句号，从而减少结尾的幻听语气音
            if i_text < len(texts) - 1:
                text += "。" if text_language != "en" else "."
        # print(i18n("实际输入的目标文本(每句):"), text)
        
        # 【修复单个字符文本的音素处理】对于单个字符文本，强制 final=True 避免自动添加点号
        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language, version, final=is_single_char)
        # print(i18n("前端处理后的文本(每句):"), norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()
        # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
        # print(cache.keys(),if_freeze)
        
        # 【修复单个字符文本拖音问题】对于单个字符文本，使用更严格的采样参数和更早的停止条件
        # 单个字符文本容易受到参考音频影响，需要降低随机性和更早停止
        if is_single_char:
            # 降低 temperature 和 top_p，使生成更稳定、更短
            single_char_temp = min(temperature, 0.5)
            single_char_top_p = min(top_p, 0.7)
            single_char_top_k = min(top_k, 15)
            # 对于单个字符，限制最大生成长度，避免拖音
            # 单个字符通常只需要很短的时间，设置更小的 early_stop_num
            single_char_max_sec = min(max_sec, 2.0)  # 最多2秒
            single_char_early_stop = hz * single_char_max_sec
        else:
            single_char_temp = temperature
            single_char_top_p = top_p
            single_char_top_k = top_k
            single_char_early_stop = hz * max_sec
        
        if(i_text in cache and if_freeze==True):pred_semantic=cache[i_text]
        else:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=single_char_top_k,
                    top_p=single_char_top_p,
                    temperature=single_char_temp,
                    early_stop_num=single_char_early_stop,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text]=pred_semantic
        t3 = ttime()
        refers=[]
        if(inp_refs):
            for path in inp_refs:
                try:
                    refer = get_spepc(hps, path.name).to(dtype).to(device)
                    refers.append(refer)
                except:
                    traceback.print_exc()
        if(len(refers)==0):refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]
        audio = (vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers,speed=speed).detach().cpu().numpy()[0, 0])
        max_audio=np.abs(audio).max()#简单防止16bit爆音
        if max_audio>1:audio/=max_audio
        
        # 【修复单个字符文本拖音问题】对于单个字符文本，检测并截断拖尾
        if is_single_char:
            # 从后往前查找，找到最后一个能量较高的位置
            # 单个字符通常很短，如果后面有很长的低能量拖尾，应该截断
            sample_rate = hps.data.sampling_rate
            # 计算音频的能量（使用滑动窗口）
            window_size = int(sample_rate * 0.05)  # 50ms窗口
            energy_threshold = 0.01  # 能量阈值
            
            # 计算每个窗口的能量
            audio_len = len(audio)
            if audio_len > window_size:
                # 从后往前找，找到最后一个能量较高的位置
                last_high_energy_idx = audio_len - 1
                for i in range(audio_len - window_size, -1, -window_size):
                    window = audio[max(0, i):min(i + window_size, audio_len)]
                    window_energy = np.abs(window).max()
                    if window_energy > energy_threshold:
                        last_high_energy_idx = min(i + window_size * 2, audio_len)  # 多保留一点
                        break
                
                # 如果检测到拖尾（最后的高能量位置距离结尾很远），截断
                tail_length = audio_len - last_high_energy_idx
                if tail_length > int(sample_rate * 0.3):  # 如果拖尾超过0.3秒
                    audio = audio[:last_high_energy_idx]
                    print(f"检测到单个字符文本拖尾，已截断 {tail_length/sample_rate:.2f} 秒")
        
        # 音量调整
        if volume != 1.0:
            audio = audio * volume
            # 再次防止爆音
            max_audio=np.abs(audio).max()
            if max_audio>1:audio/=max_audio

        # 【字母音频拼接】如果有字母音频，先添加字母音频，再添加文本音频
        if letter_audio is not None:
            # 对字母音频也应用音量调整
            letter_audio_volumed = letter_audio * volume if volume != 1.0 else letter_audio
            max_letter_audio = np.abs(letter_audio_volumed).max()
            if max_letter_audio > 1:
                letter_audio_volumed /= max_letter_audio
            audio_opt.append(letter_audio_volumed)
            # 字母音频和文本音频之间插入短暂静音（0.05秒），使衔接更自然
            short_gap = np.zeros(
                int(hps.data.sampling_rate * 0.05),
                dtype=np.float16 if is_half == True else np.float32,
            )
            audio_opt.append(short_gap)
        
        audio_opt.append(audio)
        # 只在不是最后一段时插入静音，避免末尾出现不必要的停顿
        # 同时减少静音时长，从0.15秒减少到0.1秒，使衔接更自然
        if i_text < len(texts) - 1:
            shorter_zero_wav = np.zeros(
                int(hps.data.sampling_rate * 0.1),  # 进一步缩短到0.1秒
                dtype=np.float16 if is_half == True else np.float32,
            )
            audio_opt.append(shorter_zero_wav)
        t4 = ttime()
        t.extend([t2 - t1,t3 - t2, t4 - t3])
        t1 = ttime()
    # print("%.3f\t%.3f\t%.3f\t%.3f" % 
    #        (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3]))
    #        )
    # 修正：使用 32767 并增加 clip 保护防止爆音产生的电流声
    opt_audio = np.concatenate(audio_opt, 0)
    opt_audio = np.clip(opt_audio, -1.0, 1.0)
    yield hps.data.sampling_rate, (opt_audio * 32767).astype(np.int16)


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return  "\n".join(opts)

def cut4(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split(r'(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def process_text(texts):
    _text=[]
    if all(text in [None, " ", "\n",""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    for text in texts:
        if text in  [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text


def change_choices():
    SoVITS_names, GPT_names = get_weights_names(GPT_weight_root, SoVITS_weight_root)
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


SoVITS_weight_root=["SoVITS_weights_v2","SoVITS_weights"]
GPT_weight_root=["GPT_weights_v2","GPT_weights"]
for path in SoVITS_weight_root+GPT_weight_root:
    os.makedirs(path,exist_ok=True)


def get_weights_names(GPT_weight_root, SoVITS_weight_root):
    SoVITS_names = [i for i in pretrained_sovits_name]
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (path, name))
    GPT_names = [i for i in pretrained_gpt_name]
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (path, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names(GPT_weight_root, SoVITS_weight_root)

def html_center(text, label='p'):
    return f"""<div style="text-align: center; margin: 100; padding: 50;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""

def html_left(text, label='p'):
    return f"""<div style="text-align: left; margin: 0; padding: 0;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""

# 批量生成函数
from scipy.io.wavfile import write as write_wav
import numpy as np
import time
import io
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    print("警告: ffmpeg-python 未安装，将使用传统方式（先WAV后MP3）")

# 全局变量控制批量生成
stop_batch = False
pause_batch = False

def save_audio_direct(sample_rate, audio_data, output_path, output_format="wav"):
    """
    直接从内存中的音频数据保存为指定格式（WAV或MP3）
    优化点：强制使用管道传输给 ffmpeg，避免磁盘中间 WAV 文件的写入。
    
    Args:
        sample_rate: 采样率
        audio_data: numpy数组，int16格式的音频数据
        output_path: 输出文件路径（不含扩展名）
        output_format: "wav" 或 "mp3"
    
    Returns:
        最终文件路径
    """
    if output_format == "wav":
        wav_path = f"{output_path}.wav"
        write_wav(wav_path, sample_rate, audio_data)
        return wav_path
    
    elif output_format == "mp3":
        mp3_path = f"{output_path}.mp3"
        
        # 尝试使用管道 (Pipe) 传输数据给 ffmpeg，实现零磁盘中转
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "s16le",           # 输入格式：16bit 有符号小端序 (numpy int16 的默认格式)
                "-ar", str(sample_rate), # 输入采样率
                "-ac", "1",              # 输入声道数 (模型输出通常是单声道)
                "-i", "pipe:0",          # 从 stdin 读取数据
                "-acodec", "libmp3lame",
                "-b:a", "128k",          # 目标码率
                "-ar", "44100",          # 目标采样率 (兼容性最好)
                "-ac", "2",              # 转双声道 (兼容某些对单声道支持不好的播放器)
                mp3_path,
                "-loglevel", "error"
            ]
            
            # Windows 下隐藏控制台黑窗口
            creation_flags = 0x08000000 if os.name == 'nt' else 0
            
            process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                creationflags=creation_flags
            )
            
            # 直接将内存中的字节流喂给 ffmpeg
            # 这里的 .tobytes() 几乎不产生额外开销，且避免了磁盘 I/O
            stdout, stderr = process.communicate(input=audio_data.tobytes())
            
            if process.returncode == 0 and os.path.exists(mp3_path):
                return mp3_path
            else:
                err_info = stderr.decode(errors='ignore')
                print(f"FFmpeg 管道转码失败: {err_info}")
                raise Exception(err_info)
                
        except Exception as e:
            print(f"直接管道编码失败 ({e})，回退到传统磁盘中转方式")
            # 只有当管道方式彻底失败时，才回退到写 WAV 文件再转换的老路
            wav_path = f"{output_path}.wav"
            write_wav(wav_path, sample_rate, audio_data)
            try:
                cmd = [
                    "ffmpeg", "-y", "-i", wav_path,
                    "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2", "-b:a", "128k",
                    mp3_path, "-loglevel", "error"
                ]
                subprocess.run(cmd, check=True, creationflags=creation_flags)
                if os.path.exists(mp3_path):
                    os.remove(wav_path)
                    return mp3_path
            except:
                pass
            return wav_path # 最后实在不行，返回保存好的 WAV
            
    else:
        raise ValueError(f"不支持的格式: {output_format}")

def single_inference_output(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut,
    top_k,
    top_p,
    temperature,
    ref_free,
    speed,
    if_freeze,
    inp_refs,
    volume,
    output_format,
):
    """单条生成：默认可输出 mp3；wav 时直接返回采样数据，mp3 时写入临时文件后返回路径。"""
    import tempfile

    generator = get_tts_wav(
        ref_wav_path,
        prompt_text,
        prompt_language,
        text,
        text_language,
        how_to_cut,
        top_k,
        top_p,
        temperature,
        ref_free,
        speed,
        if_freeze,
        inp_refs,
        volume,
    )
    result = None
    for res in generator:
        result = res
    if result is None:
        return None
    sample_rate, audio_data = result
    fmt = (output_format or "mp3").strip().lower()
    if fmt == "wav":
        return (sample_rate, audio_data)
    tmp_dir = tempfile.mkdtemp(prefix="tts_single_")
    base_path = os.path.join(tmp_dir, "out")
    try:
        return save_audio_direct(sample_rate, audio_data, base_path, "mp3")
    except Exception as e:
        print(f"单条生成保存 MP3 失败: {e}，回退为 WAV 数据")
        return (sample_rate, audio_data)

def batch_generation(file, name_column, text_column, data_frame, ref_wav_path, prompt_text, prompt_language, text_language, how_to_cut, top_k, top_p, temperature, ref_free, speed, if_freeze, inp_refs, volume, output_dir, output_format):
    global stop_batch, pause_batch, cache, _ref_audio_cache
    stop_batch = False
    pause_batch = False
    cache = {} # 每次批量任务开始时清理全局 cache，确保重新生成
    # 批量生成时，每行都应该使用独立的参考音频特征，避免缓存污染
    # 但为了性能，我们只在参考音频变化时才重新计算
    
    if file is None and (data_frame is None or len(data_frame) == 0):
        yield "请上传Excel文件"
        return

    # 直接使用用户在DataFrame组件中编辑过的数据
    try:
        if isinstance(data_frame, pd.DataFrame):
            df = data_frame
        else:
             df = pd.DataFrame(data_frame)
    except Exception as e:
        yield f"数据格式转换失败: {str(e)}"
        return
    
    if name_column not in df.columns:
        yield f"在表格中找不到文件名列: {name_column}"
        return
    if text_column not in df.columns:
        yield f"在表格中找不到内容列: {text_column}"
        return

    # 使用用户指定的输出目录，如果未指定则使用默认
    if not output_dir or output_dir.strip() == "":
        output_dir = "output/batch_result"
    
    # 获取绝对路径，确保用户能找到
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    total = len(df)
    results_list = [] # 用于存储生成结果
    yield f"准备开始... 结果将保存在: {abs_output_dir}", {"value": [], "__type__": "update"}
    print(f"Batch output directory: {abs_output_dir}")
    
    # 尝试清理显存
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    for index, row in df.iterrows():
        # 批量生成时，即使开启 if_freeze，也必须在每行开始时清空全局 cache，
        # 否则第 N 行会复用第 N-1 行的特征导致吃字或错乱，产生奇怪的语气词
        # 注意：函数开头已声明 global，这里直接使用即可
        cache = {}
        # 批量生成时，每行都应该独立处理，避免参考音频特征被错误复用
        # 但为了性能，如果参考音频路径相同，可以保留 prompt_semantic 缓存
        # 只清理文本相关的缓存（bert1等），确保每行文本独立处理
        if _ref_audio_cache.get("path") is not None:
            # 保留 prompt_semantic，但清理文本相关缓存
            _ref_audio_cache["bert1"] = None
            _ref_audio_cache["phones1"] = None
            _ref_audio_cache["norm_text1"] = None
        
        # 检查是否停止
        if stop_batch:
            yield f"任务已终止。已处理 {index} / {total} 条。", {"value": list(results_list), "__type__": "update"}
            return

        # 检查是否暂停
        while pause_batch:
            yield f"任务已暂停... 已处理 {index} / {total} 条。点击【继续】按钮继续。", {"value": list(results_list), "__type__": "update"}
            time.sleep(1)
            if stop_batch:
                yield f"任务已终止。已处理 {index} / {total} 条。", {"value": list(results_list), "__type__": "update"}
                return
        
        # 检查"是否处理"列 (如果存在)
        # 兼容布尔值和字符串 "True"/"False" / "是"/"否" / "t"/"f"
        is_process = True
        if "是否处理" in df.columns:
            val = row["是否处理"]
            if val is None:
                is_process = False
            elif isinstance(val, bool):
                is_process = val
            else:
                val_str = str(val).strip().lower()
                is_process = val_str in ["true", "1", "yes", "是", "on", "t"]
        
        if not is_process:
            yield f"跳过第 {index+1}/{total} 条 (未勾选)...", {"value": list(results_list), "__type__": "update"}
            continue

        # 获取文件名和文本
        filename_str = str(row[name_column]).strip()
        text = str(row[text_column]).strip()

        # 简单清洗文件名 (去除非法字符)
        filename_str = re.sub(r'[\\/*?:"<>|]', "", filename_str)

        # 判空逻辑：如果文件名或内容为空，则跳过
        if not filename_str or filename_str.lower() == "nan" or filename_str == "":
            yield f"跳过第 {index+1}/{total} 条 (文件名为空)...", {"value": list(results_list), "__type__": "update"}
            continue

        if not text or text.lower() == "nan" or text == "":
            yield f"跳过第 {index+1}/{total} 条 (合成内容为空)...", {"value": list(results_list), "__type__": "update"}
            continue
            
        yield f"正在处理第 {index+1}/{total} 条: [{filename_str}]...", {"value": list(results_list), "__type__": "update"}
        
        try:
            with torch.no_grad():
                # 批量生成时强制禁用 if_freeze，避免不同行之间的语义特征污染
                # if_freeze 主要用于单条生成时调整语速和音色，批量生成时使用会导致语气词问题
                batch_if_freeze = False
                
                # 【质量守护】限制批量生成时的采样参数，强制防止产生幻听语气词
                # 过高的参数是产生“莫名其妙语气音”的元凶
                safe_temp = min(float(temperature), 0.7)
                safe_top_p = min(float(top_p), 0.8)
                safe_top_k = min(int(top_k), 20)

                # 调用推理逻辑 (传入 volume 参数，使用安全参数，强制禁用 if_freeze)
                generator = get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, safe_top_k, safe_top_p, safe_temp, ref_free, speed, batch_if_freeze, inp_refs, volume)
                
                # 获取最后一次生成的结果
                result = None
                for res in generator:
                    result = res
            
            if result:
                sample_rate, audio_data = result
                # 使用新的直接编码函数，避免先写WAV再转换的性能损耗
                base_filename = os.path.join(abs_output_dir, filename_str)
                final_path = save_audio_direct(sample_rate, audio_data, base_filename, output_format)
                
                # 记录结果：[文件名, 内容, 音频路径]
                # Gradio 需要 "/file=" 前缀来访问本地绝对路径文件
                # 增加时间戳防止浏览器音频缓存导致预览不更新
                timestamp = int(time.time())
                audio_url = f"/file={final_path}?t={timestamp}"
                # 使用 HTML audio 标签
                audio_html = f'<audio controls src="{audio_url}"></audio>'
                
                # 第一列：待重生成勾选（默认不勾选）
                results_list.append([False, filename_str, text, audio_html])
                # 立即更新结果，确保即使只有一条数据也能看到
                yield f"已处理 {index+1}/{total} 条: [{filename_str}]", {"value": list(results_list), "__type__": "update"}

            else:
                print(f"第 {index+1} 条生成结果为空")
                # 即使生成失败，也要更新一次结果，确保界面刷新
                yield f"第 {index+1} 条生成结果为空", {"value": list(results_list), "__type__": "update"}

        except Exception as e:
            err_msg = f"处理第 {index+1} 条失败: {str(e)}"
            print(err_msg)
            import traceback
            traceback.print_exc()
            yield err_msg, {"value": list(results_list), "__type__": "update"}
            continue
            
    # 最终更新：确保所有数据都处理完成后，最后一次更新结果预览
    yield f"处理完成！文件已保存至: {abs_output_dir}", {"value": list(results_list), "__type__": "update"}

def stop_batch_task():
    global stop_batch
    stop_batch = True
    print("Batch task stopped by user.")
    return "正在终止任务..."

def pause_resume_batch_task():
    global pause_batch
    if pause_batch:
        pause_batch = False
        print("Batch task resumed.")
        return "继续生成...", "暂停"
    else:
        pause_batch = True
        print("Batch task paused.")
        return "暂停中...", "继续"

def get_excel_columns(file):
    if file is None:
        return []
    try:
        df = pd.read_excel(file.name)
        return list(df.columns)
    except:
        return []

def open_output_folder(output_dir):
    """跨平台打开输出文件夹"""
    if not output_dir or output_dir.strip() == "":
        output_dir = "output/batch_result"
    
    # 获取绝对路径
    abs_output_dir = os.path.abspath(output_dir)
    
    # 确保文件夹存在
    os.makedirs(abs_output_dir, exist_ok=True)
    
    try:
        if os.name == 'nt':  # Windows
            os.startfile(abs_output_dir)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', abs_output_dir])
        else:  # Linux
            subprocess.run(['xdg-open', abs_output_dir])
        return f"已打开文件夹: {abs_output_dir}"
    except Exception as e:
        return f"打开文件夹失败: {str(e)}\n文件夹路径: {abs_output_dir}"

# 缓存文件路径
CACHE_FILE = "./user_cache.json"

# 读取缓存
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return {}

# 保存缓存
def save_cache(gpt_path=None, sovits_path=None, ref_wav_path=None, prompt_text=None, prompt_language=None, volume=None):
    # 先加载现有缓存，然后更新
    existing_cache = load_cache()
    
    # 处理 ref_wav_path：确保是字符串路径，且文件存在
    final_ref_wav_path = None
    if ref_wav_path is not None:
        # 如果是字符串且文件存在，才保存
        if isinstance(ref_wav_path, str) and ref_wav_path.strip() and os.path.exists(ref_wav_path):
            final_ref_wav_path = ref_wav_path
        elif hasattr(ref_wav_path, 'name') and ref_wav_path.name:
            # Gradio 音频对象，检查文件是否存在
            if os.path.exists(ref_wav_path.name):
                final_ref_wav_path = ref_wav_path.name
    elif "ref_wav_path" in existing_cache:
        # 如果传入 None，检查现有缓存中的路径是否仍然有效
        existing_path = existing_cache.get("ref_wav_path")
        if existing_path and isinstance(existing_path, str) and os.path.exists(existing_path):
            final_ref_wav_path = existing_path
    
    cache = {
        "gpt_path": gpt_path if gpt_path is not None else existing_cache.get("gpt_path", ""),
        "sovits_path": sovits_path if sovits_path is not None else existing_cache.get("sovits_path", ""),
        "ref_wav_path": final_ref_wav_path,
        "prompt_text": prompt_text if prompt_text is not None else existing_cache.get("prompt_text", ""),
        "prompt_language": prompt_language if prompt_language is not None else existing_cache.get("prompt_language", i18n("中文")),
        "volume": volume if volume is not None else existing_cache.get("volume", 1.0)
    }
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")

# 加载初始值
cache_data = load_cache()
init_gpt_path = cache_data.get("gpt_path", gpt_path)
init_sovits_path = cache_data.get("sovits_path", sovits_path)
init_ref_wav_path = cache_data.get("ref_wav_path", None)
# 验证音频文件是否存在，如果不存在则设为 None，避免 Gradio 初始化错误
if init_ref_wav_path and isinstance(init_ref_wav_path, str):
    if not os.path.exists(init_ref_wav_path):
        print(f"警告：缓存中的音频文件不存在: {init_ref_wav_path}，将清空缓存")
        init_ref_wav_path = None
        # 更新缓存，移除无效的音频路径
        try:
            save_cache(ref_wav_path=None)
        except:
            pass
init_prompt_text = cache_data.get("prompt_text", "")
init_prompt_language = cache_data.get("prompt_language", i18n("中文"))
init_volume = cache_data.get("volume", 1.0)

# 如果缓存中有模型路径，优先使用缓存的（覆盖之前的环境变量读取）
if init_gpt_path and init_gpt_path in GPT_names:
    gpt_path = init_gpt_path
if init_sovits_path and init_sovits_path in SoVITS_names:
    sovits_path = init_sovits_path

# 更新模型权重
change_sovits_weights(sovits_path)
change_gpt_weights(gpt_path)

# 批量生成页：吸底操作栏 + 表格内部独立滚动（避免整页跟着滑/编辑时跳动）
_BATCH_UI_CSS = """
.batch-sticky-actions {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    background: var(--body-background-fill, var(--background-fill-primary, #fff));
    padding: 10px 16px 12px;
    border-top: 1px solid var(--border-color-primary, #e5e5e5);
    box-shadow: 0 -4px 16px rgba(0, 0, 0, 0.08);
}
/* 避免底部固定栏遮挡结果表格与操作区 */
#batch_results_df {
    margin-bottom: 72px;
}
#batch_preview_df {
    overflow: visible !important;
    overscroll-behavior: contain;
}
/* 只让表格内部（VirtualTable）滚动，外层容器不再出现第二条滚动条 */
#batch_preview_df .overflow-y-auto,
#batch_preview_df .svelte-1ipelgc {
    max-height: none !important;
    overflow: visible !important;
}
#batch_preview_df .table-wrap {
    max-height: none !important;
    overflow: hidden !important;
}
/* 生成结果表格：不限高、不滚动，按内容自动撑开 */
#batch_results_df .table-wrap,
#batch_results_df .overflow-y-auto,
#batch_results_df .svelte-1ipelgc {
    max-height: none !important;
    overflow: visible !important;
    overscroll-behavior: auto;
}
#batch_preview_df table,
#batch_results_df table {
    table-layout: fixed !important;
    width: 100% !important;
}
#batch_preview_df td,
#batch_preview_df th,
#batch_results_df td,
#batch_results_df th {
    vertical-align: top !important;
}
/* 单元格允许多行换行显示，避免长句挤成一条 */
#batch_preview_df td,
#batch_results_df td {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
    line-height: 1.45 !important;
    min-height: 2.6em;
}
#batch_preview_df td > div,
#batch_results_df td > div,
#batch_preview_df td span,
#batch_results_df td span {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
}
/* 第一列宽度固定，减少编辑引起的列宽重算跳动 */
#batch_preview_df th:first-child,
#batch_preview_df td:first-child {
    width: 90px !important;
    min-width: 90px !important;
    max-width: 90px !important;
}
#batch_results_df th:first-child,
#batch_results_df td:first-child {
    width: 100px !important;
    min-width: 100px !important;
    max-width: 100px !important;
}
#batch_preview_df input,
#batch_results_df input {
    min-height: 28px;
}
#batch_results_df {
    overscroll-behavior: contain;
}
/* 序号 / 类型 / 命名 收窄，把横向空间留给配音内容 */
#batch_preview_df th.batch-narrow-col,
#batch_preview_df td.batch-narrow-col {
    width: 88px !important;
    min-width: 72px !important;
    max-width: 110px !important;
    overflow: hidden;
    white-space: nowrap !important;
}
#batch_preview_df th.batch-name-col,
#batch_preview_df td.batch-name-col {
    width: 128px !important;
    min-width: 100px !important;
    max-width: 150px !important;
    overflow: hidden;
    white-space: nowrap !important;
}
/* 配音内容列：外观接近多行文本框，占剩余大部分宽度 */
#batch_preview_df th.batch-text-cell,
#batch_preview_df td.batch-text-cell {
    min-width: 360px !important;
    width: 62% !important;
}
#batch_preview_df td.batch-text-cell .cell-wrap {
    display: block;
    width: 100%;
    position: relative;
}
#batch_preview_df td.batch-text-cell span.edit {
    display: none !important;
}
#batch_preview_df td.batch-text-cell span:not(.edit) {
    display: block;
    min-height: 0;
    padding: 6px 8px !important;
    border: 1px solid var(--input-border-color, #c5c5d2);
    border-radius: 6px;
    background: var(--input-background-fill, #fff);
    line-height: 1.5;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}
#batch_preview_df textarea.batch-cell-textarea {
    display: block;
    width: 100%;
    min-height: 36px;
    max-height: none;
    height: auto;
    padding: 6px 8px;
    margin: 0;
    box-sizing: border-box;
    resize: none;
    overflow: hidden;
    line-height: 1.5;
    font-size: 14px;
    font-family: inherit;
    color: inherit;
    white-space: pre-wrap;
    word-break: break-word;
    border: 1px solid var(--color-accent, #2563eb);
    border-radius: 6px;
    background: var(--input-background-fill, #fff);
    outline: none;
}
"""

_BATCH_UI_HEAD = """
<script>
(function () {
  function headerText(el) {
    return String(el && el.textContent || "").replace(/\\s+/g, "");
  }
  function isTextHeader(text) {
    var t = headerText({ textContent: text });
    return t.indexOf("配音内容") !== -1 || t.indexOf("文本内容") !== -1 || t.indexOf("合成内容") !== -1;
  }
  function colClassForHeader(text) {
    var t = headerText({ textContent: text });
    if (isTextHeader(t)) return "batch-text-cell";
    if (t.indexOf("命名") !== -1 || t.indexOf("文件名") !== -1) return "batch-name-col";
    if (t.indexOf("序号") !== -1 || t.indexOf("类型") !== -1) return "batch-narrow-col";
    return "";
  }
  function markTextCells(root) {
    var ths = Array.prototype.slice.call(root.querySelectorAll("thead th"));
    if (!ths.length) ths = Array.prototype.slice.call(root.querySelectorAll("table tr:first-child th"));
    var classes = ths.map(function (th) { return colClassForHeader(th.textContent); });
    var hasText = classes.some(function (c) { return c === "batch-text-cell"; });
    if (!hasText && ths.length > 1) classes[ths.length - 1] = "batch-text-cell";
    ths.forEach(function (th, i) {
      if (classes[i]) th.classList.add(classes[i]);
    });
    Array.prototype.forEach.call(root.querySelectorAll("tbody tr"), function (tr) {
      classes.forEach(function (cls, i) {
        if (cls && tr.children[i]) tr.children[i].classList.add(cls);
      });
    });
  }
  function fitTextarea(ta) {
    if (!ta) return;
    ta.style.height = "auto";
    var next = ta.scrollHeight;
    if (next < 36) next = 36;
    ta.style.height = next + "px";
    ta.style.overflow = "hidden";
  }
  function upgradeTextarea(input) {
    if (!input || input.dataset.batchTa === "1") return;
    var td = input.closest("td");
    if (!td || !td.classList.contains("batch-text-cell")) return;
    input.dataset.batchTa = "1";
    var wrap = input.parentNode;
    var ta = wrap.querySelector("textarea.batch-cell-textarea");
    if (!ta) {
      ta = document.createElement("textarea");
      ta.className = "batch-cell-textarea";
      ta.rows = 1;
      wrap.insertBefore(ta, input);
    }
    var span = wrap.querySelector("span");
    var recovered = input.value;
    if (!recovered && span && span.textContent) recovered = span.textContent;
    ta.value = recovered || "";
    if (input.value !== ta.value) {
      input.value = ta.value;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
    input.setAttribute("tabindex", "-1");
    input.style.cssText = "position:absolute;left:0;top:0;width:1px;height:1px;opacity:0;pointer-events:none;";
    if (!ta.dataset.bound) {
      ta.dataset.bound = "1";
      ta.addEventListener("input", function () {
        input.value = ta.value;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        fitTextarea(ta);
      });
      ta.addEventListener("blur", function () {
        input.value = ta.value;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
        input.dispatchEvent(new Event("blur", { bubbles: true }));
      });
      ta.addEventListener("keydown", function (e) {
        e.stopPropagation();
      });
    }
    setTimeout(function () {
      fitTextarea(ta);
      ta.focus();
    }, 0);
  }
  function scan(root) {
    if (!root) return;
    markTextCells(root);
    Array.prototype.forEach.call(root.querySelectorAll("td.batch-text-cell input:not([type=checkbox])"), upgradeTextarea);
  }
  function boot() {
    var obs = new MutationObserver(function () {
      var root = document.getElementById("batch_preview_df");
      if (root) scan(root);
    });
    obs.observe(document.body, { childList: true, subtree: true });
    document.addEventListener("keydown", function (e) {
      if (e.key.length !== 1 || e.ctrlKey || e.metaKey || e.altKey) return;
      var td = e.target && e.target.closest ? e.target.closest("#batch_preview_df td.batch-text-cell") : null;
      if (!td) return;
      if (e.target.tagName === "TEXTAREA") return;
      e.stopPropagation();
    }, true);
    var root = document.getElementById("batch_preview_df");
    if (root) scan(root);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
</script>
"""

with gr.Blocks(title="GPT-SoVITS WebUI", css=_BATCH_UI_CSS, head=_BATCH_UI_HEAD) as app:
    gr.Markdown(
        value=i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
    )
    with gr.Group():
        gr.Markdown(html_center(i18n("模型切换"),'h3'))
        with gr.Row():
            GPT_dropdown = gr.Dropdown(label=i18n("GPT模型列表"), choices=sorted(GPT_names, key=custom_sort_key), value=gpt_path, interactive=True, scale=14)
            SoVITS_dropdown = gr.Dropdown(label=i18n("SoVITS模型列表"), choices=sorted(SoVITS_names, key=custom_sort_key), value=sovits_path, interactive=True, scale=14)
            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary", scale=14)
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
        gr.Markdown(html_center(i18n("*请上传并填写参考信息"),'h3'))
        with gr.Row():
            inp_ref = gr.Audio(label=i18n("请上传3~10秒内参考音频，超过会报错！"), type="filepath", scale=13, value=init_ref_wav_path)
            with gr.Column(scale=13):
                ref_text_free = gr.Checkbox(label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。"), value=False, interactive=True, show_label=True,scale=1)
                gr.Markdown(html_left(i18n("使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥(不晓得写啥)可以开。<br>开启后无视填写的参考文本。")))
                prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value=init_prompt_text, lines=5, max_lines=5,scale=1)
            with gr.Column(scale=14):
                prompt_language = gr.Dropdown(
                    label=i18n("参考音频的语种"), choices=list(dict_language.keys()), value=init_prompt_language,
                )
                inp_refs = gr.File(label=i18n("可选项：通过拖拽多个文件上传多个参考音频（建议同性），平均融合他们的音色。如不填写此项，音色由左侧单个参考音频控制。如是微调模型，建议参考音频全部在微调训练集音色内，底模不用管。"),file_count="multiple")
        
        # 移动音量控制到这里 (参考信息下方，所有模式共享)
        with gr.Row():
             volume = gr.Slider(minimum=0.1, maximum=2.0, step=0.05, label="全局音量调节 (1.0为原音量)", value=init_volume, scale=4)
             volume_preview = gr.Audio(label="音量试听 (仅截取前5秒)", interactive=False, scale=4)
             # 【修复 Gradio 4.x 兼容性】使用 gr.Column(scale=...) 替代 gr.Markdown(scale=...)
             with gr.Column(scale=6):
                 pass 

        # 定义试听函数 (移到这里)
        def preview_volume(ref_wav_path, vol):
            if not ref_wav_path:
                return None
            
            # 处理 ref_wav_path：如果是 Gradio 音频对象，提取文件路径
            if isinstance(ref_wav_path, str):
                audio_path = ref_wav_path
            elif hasattr(ref_wav_path, 'name'):  # Gradio 音频对象
                audio_path = ref_wav_path.name if ref_wav_path.name else None
            else:
                audio_path = str(ref_wav_path) if ref_wav_path else None
            
            if not audio_path or not os.path.exists(audio_path):
                return None
            
            try:
                wav, sr = librosa.load(audio_path, sr=None)
                if len(wav) == 0:
                    return None
                max_len = 5 * sr
                if len(wav) > max_len:
                    wav = wav[:max_len]
                wav = wav * vol
                max_val = np.abs(wav).max()
                if max_val > 1:
                    wav = wav / max_val
                return (sr, (wav * 32767).astype(np.int16))
            except Exception as e:
                print(f"Error previewing volume: {e}")
                import traceback
                traceback.print_exc()
                return None

        # 绑定试听事件
        volume.release(preview_volume, [inp_ref, volume], [volume_preview])
        
        # 绑定保存缓存的事件（包括音量）
        def save_cache_with_volume(gpt_path, sovits_path, ref_wav_path, prompt_text, prompt_language, vol):
            # 处理 ref_wav_path：如果是 Gradio 音频对象，提取文件路径；如果是字符串，直接使用；如果是 None，保持 None
            ref_path = None
            if ref_wav_path is not None:
                if isinstance(ref_wav_path, str) and ref_wav_path.strip():
                    # 字符串路径，检查文件是否存在
                    if os.path.exists(ref_wav_path):
                        ref_path = ref_wav_path
                elif hasattr(ref_wav_path, 'name') and ref_wav_path.name:  # Gradio 音频对象
                    # Gradio 音频对象，提取文件路径并检查是否存在
                    if os.path.exists(ref_wav_path.name):
                        ref_path = ref_wav_path.name
            
            # 确保 prompt_text 是字符串
            prompt_txt = str(prompt_text) if prompt_text is not None else ""
            
            # 确保 prompt_language 是字符串
            prompt_lang = str(prompt_language) if prompt_language is not None else i18n("中文")
            
            try:
                save_cache(gpt_path=gpt_path, sovits_path=sovits_path, ref_wav_path=ref_path, 
                          prompt_text=prompt_txt, prompt_language=prompt_lang, volume=vol)
            except Exception as e:
                print(f"保存缓存时出错: {e}")
                import traceback
                traceback.print_exc()
        
        for comp in [GPT_dropdown, SoVITS_dropdown, inp_ref, prompt_text, prompt_language]:
            comp.change(save_cache_with_volume, [GPT_dropdown, SoVITS_dropdown, inp_ref, prompt_text, prompt_language, volume], [])
        
        # 音量单独绑定保存事件
        volume.change(lambda vol: save_cache(volume=vol), [volume], [])

        gr.Markdown(html_center(i18n("*请填写需要合成的目标文本和语种模式"),'h3'))
        
        with gr.Tabs():
            with gr.TabItem(i18n("单条生成")):
                with gr.Row():
                    with gr.Column(scale=13):
                        text = gr.Textbox(label=i18n("需要合成的文本"), value="", lines=26, max_lines=26)
                    with gr.Column(scale=7):
                        text_language = gr.Dropdown(
                                label=i18n("需要合成的语种")+i18n(".限制范围越小判别效果越好。"), choices=list(dict_language.keys()), value=i18n("中文"), scale=1
                            )
                        how_to_cut = gr.Dropdown(
                                label=i18n("怎么切"),
                                choices=[i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ],
                                value=i18n("按标点符号切"),
                                interactive=True, scale=1
                            )
                        gr.Markdown(value=html_center(i18n("语速调整，高为更快")))
                        if_freeze=gr.Checkbox(label=i18n("是否直接对上次合成结果调整语速和音色。防止随机性。"), value=False, interactive=True,show_label=True, scale=1)
                        speed = gr.Slider(minimum=0.6,maximum=1.65,step=0.05,label=i18n("语速"),value=1,interactive=True, scale=1)
                        gr.Markdown(html_center(i18n("GPT采样参数(无参考文本时不要太低。不懂就用默认)：")))
                        top_k = gr.Slider(minimum=1,maximum=100,step=1,label=i18n("top_k"),value=15,interactive=True, scale=1)
                        top_p = gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("top_p"),value=1,interactive=True, scale=1)
                        temperature = gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("temperature"),value=1,interactive=True,  scale=1)
                        single_output_format = gr.Radio(
                            label=i18n("输出格式"),
                            choices=["mp3", "wav"],
                            value="mp3",
                            scale=1,
                        )
                    # with gr.Column():
                    #     gr.Markdown(value=i18n("手工调整音素。当音素框不为空时使用手工音素输入推理，无视目标文本框。"))
                    #     phoneme=gr.Textbox(label=i18n("音素框"), value="")
                    #     get_phoneme_button = gr.Button(i18n("目标文本转音素"), variant="primary")
                with gr.Row():
                    inference_button = gr.Button(i18n("合成语音"), variant="primary", size='lg', scale=25)
                    output = gr.Audio(label=i18n("输出的语音"), scale=14)

                inference_button.click(
                    single_inference_output,
                    [
                        inp_ref,
                        prompt_text,
                        prompt_language,
                        text,
                        text_language,
                        how_to_cut,
                        top_k,
                        top_p,
                        temperature,
                        ref_text_free,
                        speed,
                        if_freeze,
                        inp_refs,
                        volume,
                        single_output_format,
                    ],
                    [output],
                )
            
            with gr.TabItem(i18n("批量生成")):
                gr.Markdown(html_center(i18n("批量生成模式"),'h3'))
                # 调整为一行四列等宽布局
                with gr.Row():
                    batch_file = gr.File(label="1. 上传Excel文件 (.xlsx)", file_types=[".xlsx"], scale=1)
                    batch_sheet = gr.Dropdown(label="2. 选择工作表", choices=[], interactive=True, scale=1)
                    batch_name_col = gr.Dropdown(label="3. 选择文件名所在列", choices=[], interactive=True, scale=1)
                    batch_text_col = gr.Dropdown(label="4. 选择合成内容所在列", choices=[], interactive=True, scale=1)
                
                # 3. 可编辑表格 (单独一行)
                with gr.Row():
                    gr.Markdown("### 3. 编辑/预览数据 (第一列【是否处理】可控制是否生成该行)")
                
                # (删除了原先在这里的音量滑块和试听组件，因为已经移到全局区域了)

                # 添加批量操作按钮
                with gr.Row():
                    btn_check_all = gr.Button("全选 (全部处理)", variant="secondary", size="sm", scale=1)
                    btn_uncheck_all = gr.Button("取消全选 (暂不处理)", variant="secondary", size="sm", scale=1)
                    # 【修复 Gradio 4.x 兼容性】使用 gr.Column(scale=...) 替代 gr.Markdown(scale=...)
                    with gr.Column(scale=8):
                        pass
                
                # 初始化时，先用一个空的结构
                # 【关键修复】预设一个足够长的 datatype 列表，确保第一列始终被识别为 bool (Checkbox)
                # 后面预设 100 列为 str，通常够用了。Gradio 会根据这个配置渲染对应列。
                # 这样即使不能动态更新 datatype，也能利用初始化的配置。
                DEFAULT_DATATYPE = ["bool"] + ["str"] * 100
                
                batch_preview = gr.DataFrame(
                    label="完整数据预览与编辑（双击「配音内容」单元格可用多行输入框修改）", 
                    value=[[True, "示例文件名", "示例内容"]], 
                    headers=["是否处理", "文件名", "文本内容"], 
                    interactive=True,
                    type="pandas",
                    wrap=True,
                    height=520,
                    elem_id="batch_preview_df",
                    # col_count=(3, "dynamic"), 
                    datatype=DEFAULT_DATATYPE 
                )

                def _to_process_bool(val):
                    """将「是否处理」统一为 bool，避免字符串 f/t 编辑引发表格跳动。"""
                    if isinstance(val, (bool, np.bool_)):
                        return bool(val)
                    if val is None:
                        return False
                    s = str(val).strip().lower()
                    return s in ("true", "1", "yes", "是", "on", "t")

                # 批量修改状态的函数
                def set_all_status(df, status):
                    if df is None or df.empty:
                        return gr.update()
                    try:
                        # 确保第一列存在
                        if len(df.columns) > 0:
                            # 修改第一列的所有值为 status (True/False)
                            df.iloc[:, 0] = bool(status)
                        return gr.update(value=df)
                    except Exception as e:
                        print(f"Error updating dataframe: {e}")
                        return gr.update()

                # 绑定按钮事件
                btn_check_all.click(lambda df: set_all_status(df, True), [batch_preview], [batch_preview])
                btn_uncheck_all.click(lambda df: set_all_status(df, False), [batch_preview], [batch_preview])

                # 核心读取逻辑提取
                def read_excel_with_sheet(file, sheet_name=None):
                    if file is None:
                        return None, [], [], "请先上传文件"
                    
                    try:
                        # 1. 获取所有可见 Sheet 名称
                        # openpyxl 引擎可以读取 sheet 的状态（visible/hidden）
                        wb = openpyxl.load_workbook(file.name, read_only=True)
                        visible_sheets = []
                        for sheet in wb:
                            if sheet.sheet_state == 'visible':
                                visible_sheets.append(sheet.title)
                        
                        sheet_names = visible_sheets if visible_sheets else wb.sheetnames # 如果没有可见的（这不太可能），就回退到所有
                        wb.close()
                        
                        # 2. 确定要读取的 Sheet
                        target_sheet = sheet_name
                        if target_sheet is None or target_sheet not in sheet_names:
                            target_sheet = sheet_names[0]
                        
                        # 3. 读取特定 Sheet
                        df = pd.read_excel(file.name, sheet_name=target_sheet)
                        
                        # 4. 数据清洗
                        df.dropna(axis=1, how='all', inplace=True)
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        df.dropna(how='all', inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        
                        # 5. 插入/规范化状态列（统一为 bool，兼容 Excel 中的 t/f 等写法）
                        if "是否处理" not in df.columns:
                            df.insert(0, "是否处理", True)
                        else:
                            df["是否处理"] = df["是否处理"].map(_to_process_bool)
                        
                        # 6. 类型转换
                        for col in df.columns[1:]:
                            df[col] = df[col].fillna("").astype(str)
                            
                        cols = list(df.columns)
                        return df, cols, sheet_names, target_sheet
                        
                    except Exception as e:
                        print(f"Error reading excel: {e}")
                        return None, [], [], str(e)

                # 智能选列逻辑提取
                def smart_select_cols(cols):
                    val_name = None
                    val_text = None
                    if "命名" in cols: val_name = "命名"
                    if "配音内容" in cols: val_text = "配音内容"
                    if val_name is None: val_name = cols[1] if len(cols) > 1 else None
                    if val_text is None: val_text = cols[2] if len(cols) > 2 else (cols[1] if len(cols) > 1 else None)
                    return val_name, val_text

                # 处理文件上传
                def handle_file_upload(file):
                    df, cols, sheets, target_sheet = read_excel_with_sheet(file)
                    
                    if df is None:
                        return (
                            gr.update(value=pd.DataFrame()), 
                            gr.update(choices=[], value=None), # Sheet
                            gr.update(choices=[], value=None), # Name
                            gr.update(choices=[], value=None), # Text
                            gr.update(value="output/batch_result"), # Dir
                            f"导入失败: {target_sheet}" # Error msg
                        )
                    
                    val_name, val_text = smart_select_cols(cols)
                    msg = f"成功导入工作表: {target_sheet} (共 {len(df)} 条)"
                    
                    # 使用工作表名作为输出文件夹名（如：output/总表）
                    try:
                        safe_sheet_name = str(target_sheet).strip()
                        safe_sheet_name = re.sub(r'[\\/*?:"<>|]', "_", safe_sheet_name)
                        new_output_dir = f"output/{safe_sheet_name}" if safe_sheet_name else "output/batch_result"
                    except:
                        new_output_dir = "output/batch_result"

                    return (
                        gr.update(value=df),
                        gr.update(choices=sheets, value=target_sheet),
                        gr.update(choices=cols, value=val_name),
                        gr.update(choices=cols, value=val_text),
                        gr.update(value=new_output_dir),
                        msg
                    )

                # 处理 Sheet 切换
                def handle_sheet_change(file, sheet_name):
                    # 如果只是点了一下 Dropdown 但没选值，或者文件空了，直接返回
                    if file is None or sheet_name is None:
                        return gr.update(), gr.update(), gr.update(), gr.update(), "请选择有效的工作表"

                    df, cols, sheets, target_sheet = read_excel_with_sheet(file, sheet_name)
                    
                    if df is None:
                        return gr.update(), gr.update(), gr.update(), gr.update(), f"切换失败: {target_sheet}"
                        
                    val_name, val_text = smart_select_cols(cols)
                    msg = f"已切换至工作表: {target_sheet} (共 {len(df)} 条)"
                    try:
                        safe_sheet_name = str(target_sheet).strip()
                        safe_sheet_name = re.sub(r'[\\/*?:"<>|]', "_", safe_sheet_name)
                        new_output_dir = f"output/{safe_sheet_name}" if safe_sheet_name else "output/batch_result"
                    except:
                        new_output_dir = "output/batch_result"
                    
                    return (
                        gr.update(value=df),
                        gr.update(choices=cols, value=val_name),
                        gr.update(choices=cols, value=val_text),
                        gr.update(value=new_output_dir),
                        msg
                    )

                # 4. 操作区
                with gr.Row():
                    gr.Markdown("### 4. 操作与输出")

                with gr.Row():
                    batch_output_dir = gr.Textbox(label="输出文件夹路径 (留空则默认为 output/batch_result)", value="output/batch_result", scale=3)
                    btn_open_folder = gr.Button("📁 打开文件夹", variant="secondary", scale=1)
                    batch_output_format = gr.Radio(label="输出格式", choices=["wav", "mp3"], value="mp3", scale=1)
                
                batch_status = gr.Textbox(label="处理状态")

                # 绑定打开文件夹按钮
                btn_open_folder.click(open_output_folder, [batch_output_dir], [batch_status])
                
                # 5. 生成结果展示
                with gr.Row():
                    gr.Markdown("### 5. 生成结果预览 (实时更新，点击音频即可播放)")
                
                batch_results_df = gr.DataFrame(
                    label="生成结果",
                    headers=["待重生成", "文件名", "文本内容", "音频预览"],
                    datatype=["bool", "str", "str", "html"],
                    interactive=True,
                    wrap=True,
                    elem_id="batch_results_df",
                    column_widths=["100px", "160px", "35%", "30%"],
                )
                
                with gr.Row():
                    gr.Markdown(
                        "在「待重生成」列勾选不满意的行，试听结束后点击「批量重新生成勾选行」统一重做；重新生成后仅展示本次成功重做的结果。「清除勾选」可一键取消所有勾选。"
                    )

                # 操作按钮吸底：生成 / 暂停 / 终止 + 重新生成 / 清除勾选
                with gr.Row(elem_id="batch_action_bar", elem_classes=["batch-sticky-actions"]):
                    batch_btn = gr.Button("开始批量生成", variant="primary", scale=2)
                    batch_pause = gr.Button("暂停", variant="secondary", scale=1)
                    batch_stop = gr.Button("终止", variant="stop", scale=1)
                    btn_regen_checked = gr.Button("批量重新生成勾选行", variant="primary", scale=2)
                    btn_clear_regen_marks = gr.Button("清除勾选", variant="secondary", scale=1)

                def _batch_result_row_marked(val):
                    """解析「待重生成」列是否为勾选。"""
                    if isinstance(val, (bool, np.bool_)):
                        return bool(val)
                    if val is None:
                        return False
                    s = str(val).strip().lower()
                    return s in ("true", "1", "yes", "是", "on", "t")

                def _normalize_batch_results_df(df):
                    """兼容旧版三列表格：自动插入「待重生成」列。"""
                    if df is None or (hasattr(df, "empty") and df.empty):
                        return df
                    try:
                        n = len(df.columns)
                    except Exception:
                        return df
                    if n == 3:
                        out = df.copy()
                        out.insert(0, "待重生成", False)
                        return out
                    return df

                def clear_regen_checkboxes(df):
                    df = _normalize_batch_results_df(df)
                    if df is None or df.empty:
                        return gr.update()
                    df2 = df.copy()
                    for i in range(len(df2)):
                        df2.iloc[i, 0] = False
                    return gr.update(value=df2)

                def regenerate_checked_rows(df, ref_wav_path, prompt_text, prompt_language, text_language, how_to_cut, top_k, top_p, temperature, ref_free, speed, if_freeze, inp_refs, volume, output_dir, output_format):
                    df = _normalize_batch_results_df(df)
                    if df is None or df.empty:
                        return "没有可重新生成的数据", {"value": df if df is not None else pd.DataFrame(), "__type__": "update"}
                    if len(df.columns) < 4:
                        return "结果表格式异常，请重新执行一次批量生成。", {"value": df, "__type__": "update"}

                    marked_indices = [i for i in range(len(df)) if _batch_result_row_marked(df.iloc[i, 0])]
                    if not marked_indices:
                        return "请先在「待重生成」列勾选需要重做的行。", {"value": df, "__type__": "update"}

                    global cache
                    if not output_dir or str(output_dir).strip() == "":
                        output_dir = "output/batch_result"
                    abs_output_dir = os.path.abspath(output_dir)
                    os.makedirs(abs_output_dir, exist_ok=True)

                    # 重新生成后仅保留本次成功重做的结果，未勾选/未重做的行不再展示
                    regenerated_rows = []
                    ok_names = []
                    err_msgs = []

                    for idx in marked_indices:
                        cache = {}
                        filename = str(df.iloc[idx, 1]).strip()
                        text = str(df.iloc[idx, 2]).strip()
                        if not filename or not text:
                            err_msgs.append(f"第 {idx + 1} 行: 文件名或文本为空，已跳过")
                            continue
                        try:
                            safe_temp = min(float(temperature), 0.7)
                            safe_top_p = min(float(top_p), 0.8)
                            safe_top_k = min(int(top_k), 20)
                            batch_if_freeze = False
                            with torch.no_grad():
                                generator = get_tts_wav(
                                    ref_wav_path,
                                    prompt_text,
                                    prompt_language,
                                    text,
                                    text_language,
                                    how_to_cut,
                                    safe_top_k,
                                    safe_top_p,
                                    safe_temp,
                                    ref_free,
                                    speed,
                                    batch_if_freeze,
                                    inp_refs,
                                    volume,
                                )
                                result = None
                                for res in generator:
                                    result = res
                            if result:
                                sample_rate, audio_data = result
                                base_filename = os.path.join(abs_output_dir, filename)
                                final_path = save_audio_direct(sample_rate, audio_data, base_filename, output_format)
                                timestamp = int(time.time())
                                audio_url = f"/file={final_path}?t={timestamp}"
                                audio_html = f'<audio controls src="{audio_url}"></audio>'
                                regenerated_rows.append([False, filename, text, audio_html])
                                ok_names.append(filename)
                            else:
                                err_msgs.append(f"{filename}: 生成结果为空")
                        except Exception as e:
                            print(f"Regenerate error row {idx}: {e}")
                            import traceback
                            traceback.print_exc()
                            err_msgs.append(f"{filename}: {str(e)}")

                    parts = []
                    if ok_names:
                        parts.append(f"已重新生成 {len(ok_names)} 条: " + "、".join(ok_names[:20]) + ("…" if len(ok_names) > 20 else ""))
                        parts.append("结果预览仅显示本次重生成条目")
                    if err_msgs:
                        parts.append("部分失败: " + "; ".join(err_msgs[:10]) + ("…" if len(err_msgs) > 10 else ""))
                    msg = " | ".join(parts) if parts else "未处理任何行"
                    return msg, {"value": regenerated_rows, "__type__": "update"}

                btn_regen_checked.click(
                    regenerate_checked_rows,
                    [batch_results_df, inp_ref, prompt_text, prompt_language, text_language, how_to_cut, top_k, top_p, temperature, ref_text_free, speed, if_freeze, inp_refs, volume, batch_output_dir, batch_output_format],
                    [batch_status, batch_results_df],
                )
                btn_clear_regen_marks.click(
                    clear_regen_checkboxes,
                    [batch_results_df],
                    [batch_results_df],
                )

                batch_file.change(handle_file_upload, [batch_file], [batch_preview, batch_sheet, batch_name_col, batch_text_col, batch_output_dir, batch_status])
                batch_sheet.change(handle_sheet_change, [batch_file, batch_sheet], [batch_preview, batch_name_col, batch_text_col, batch_output_dir, batch_status])

                batch_btn.click(
                    batch_generation,
                    [batch_file, batch_name_col, batch_text_col, batch_preview, inp_ref, prompt_text, prompt_language, text_language, how_to_cut, top_k, top_p, temperature, ref_text_free, speed, if_freeze, inp_refs, volume, batch_output_dir, batch_output_format],
                    [batch_status, batch_results_df] # 更新两个输出：状态文本和结果表格
                )
                batch_stop.click(stop_batch_task, [], [batch_status, batch_results_df], queue=False)
                batch_pause.click(pause_resume_batch_task, [], [batch_status, batch_pause, batch_results_df], queue=False)

        SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown,prompt_language,text_language], [prompt_language,text_language,prompt_text,prompt_language,text,text_language])
        GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])

if __name__ == '__main__':
    # 自动查找可用端口（如果指定端口被占用，尝试下一个端口）
    import socket
    def find_free_port(start_port, max_attempts=10):
        """从 start_port 开始查找可用端口"""
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', port))
                    return port
            except OSError:
                continue
        # 如果都不可用，返回 None 让 Gradio 自动选择
        print(f"警告: 端口 {start_port} 到 {start_port + max_attempts - 1} 都被占用，将使用随机端口")
        return None
    
    actual_port = find_free_port(infer_ttswebui)
    if actual_port and actual_port != infer_ttswebui:
        print(f"端口 {infer_ttswebui} 被占用，自动切换到端口 {actual_port}")

    def list_lan_ips():
        ips = []
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if ip and not ip.startswith("127.") and ip not in ips:
                    ips.append(ip)
        except Exception:
            pass
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.insert(0, ip)
        except Exception:
            pass
        return ips

    # 监听所有网卡，方便同一局域网里的其他 Mac 访问（不要绑 127.0.0.1）
    launch_server_name = "0.0.0.0"
    launch_inbrowser = True
    launch_port = actual_port if actual_port else infer_ttswebui
    lan_ips = list_lan_ips()
    print("WebUI 访问地址：")
    print(f"  本机: http://127.0.0.1:{launch_port}")
    if lan_ips:
        for ip in lan_ips:
            print(f"  其他电脑/Mac（同一局域网）: http://{ip}:{launch_port}")
    else:
        print("  未自动识别局域网 IP，请在运行服务的电脑上看系统网络设置里的 IP。")
    print("  若两台电脑不在同一网络，请使用启动后给出的 gradio.live 公网地址。")

    launch_kwargs = dict(
        server_name=launch_server_name,
        inbrowser=launch_inbrowser,
        share=is_share,
        server_port=launch_port,
        quiet=True,
        allowed_paths=["output/batch_result", "."] # 允许访问当前目录及输出目录
    )

    try:
        app.queue().launch(**launch_kwargs)  # concurrency_count=511, max_size=1022
    except ValueError as e:
        err_msg = str(e)
        if "localhost is not accessible" in err_msg and not launch_kwargs["share"]:
            print("检测到 localhost 访问受限，自动切换为 share=True 重试启动。")
            launch_kwargs["share"] = True
            app.queue().launch(**launch_kwargs)
        else:
            raise
