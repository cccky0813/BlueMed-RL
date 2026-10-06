"""Expert Agent 的影像工具：FastSAM 分割（本地）+ 视觉大模型结构化描述。

安全设计（工具沙箱 / 最小权限）：模型不能传文件路径，只能传 image_id（格式 img_ + 8 位十六进制）；
工具只在 data/uploads/ 里查找这个编号对应的文件，防止路径穿越读取任意文件。
"""

import base64
import re
import sys
import uuid
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")    # FastSAM 用 matplotlib 画分割图；Web 服务里请求跑在子线程，macOS 的 GUI 后端只能在主线程建窗口会直接崩溃，
                         # 所以必须用无界面的 Agg 后端（只在内存里画图）
import numpy as np
from langchain_core.tools import tool
from PIL import Image
from pydantic import BaseModel, Field

from bluemed import config
from bluemed.llm import get_llm
from bluemed.schemas import Evidence, ImageFindings

UPLOAD_DIR = config.ROOT / "data" / "uploads"
SEGMENT_DIR = config.ROOT / "data" / "segmented"
IMAGE_ID = re.compile(r"^img_[0-9a-f]{8}$")

VISION_PROMPT = """你是医学影像助手。左图是原始医学影像，右图是 FastSAM 分割结果（不同颜色代表不同区域）。
只描述能看到的客观表现，不要下诊断。输出 JSON，字段：
modality（影像类型）、body_part（部位）、findings（可见表现，最多 5 条）、keywords（可用于检索医学知识图谱的疾病或症状词，最多 5 个）。"""


def save_upload(data: bytes) -> str:
    """保存用户上传的图片，返回 image_id。统一转成 PNG，丢弃原文件名（不信任用户提供的文件名）。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    image_id = f"img_{uuid.uuid4().hex[:8]}"
    Image.open(__import__("io").BytesIO(data)).convert("RGB").save(UPLOAD_DIR / f"{image_id}.png")
    return image_id


def resolve(image_id: str) -> Path | None:
    if not IMAGE_ID.match(image_id or ""):
        return None
    path = (UPLOAD_DIR / f"{image_id}.png").resolve()
    return path if path.parent == UPLOAD_DIR.resolve() and path.exists() else None


@lru_cache(maxsize=1)
def _fastsam():
    sys.path.insert(0, str(config.ROOT / "third_party"))
    import torch
    from fastsam import FastSAM
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return FastSAM(str(config.ROOT / "weights" / "FastSAM_X.pt")), device


def segment(path: Path, size: int = 1024) -> tuple[Path, int]:
    """FastSAM 全图分割，返回（分割图路径，区域数）。"""
    model, device = _fastsam()                     # 先加载（同时把 third_party 加入导入路径），再导入 FastSAMPrompt
    from fastsam import FastSAMPrompt
    image = Image.open(path).convert("RGB")
    scale = size / max(image.size)
    image = image.resize((int(image.width * scale), int(image.height * scale)))
    results = model(image, device=device, retina_masks=True, iou=0.7, conf=0.25, imgsz=size, verbose=False)
    prompt = FastSAMPrompt(image, results, device=device)
    masks = prompt.everything_prompt()
    plotted = prompt.plot_to_result(annotations=masks, mask_random_color=True, retina=True, withContours=True)
    if plotted.dtype != np.uint8:
        plotted = (plotted * 255).astype(np.uint8)
    SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
    out = SEGMENT_DIR / f"{path.stem}_seg.png"
    Image.fromarray(plotted).save(out)
    return out, (len(masks) if masks is not None else 0)


def _b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def describe(original: Path, segmented: Path) -> ImageFindings:
    """原图 + 分割图一起交给视觉大模型，按 schema 输出结构化描述（json_mode + Pydantic 校验）。"""
    llm = get_llm(thinking=False).bind(response_format={"type": "json_object"})
    reply = llm.invoke([("user", [
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": _b64(original)}},
        {"type": "image_url", "image_url": {"url": _b64(segmented)}},
    ])])
    return ImageFindings.model_validate_json(reply.content)


class ImageArgs(BaseModel):
    image_id: str = Field(description="用户上传图片的编号，格式 img_xxxxxxxx，问题中会给出")


@tool(args_schema=ImageArgs, response_format="content_and_artifact")
def analyze_image(image_id: str):
    """分析用户上传的医学影像：先用 FastSAM 分割，再由视觉大模型描述影像类型、部位、可见表现，并给出可检索图谱的关键词。
    用户上传了图片时先调用它，再用返回的关键词调用图谱工具查询相关知识。"""
    path = resolve(image_id)
    if path is None:
        return f"找不到图片 {image_id}（只能分析用户本次上传的图片）。", []
    seg_path, n_masks = segment(path)
    f = describe(path, seg_path)
    evidence = [Evidence.make("image", image_id, "影像类型与部位", f"{f.modality}，{f.body_part}")]
    evidence += [Evidence.make("image", image_id, "影像所见", x) for x in f.findings]
    text = "\n".join(e.to_text() for e in evidence)
    return f"{text}\nFastSAM 分割出 {n_masks} 个区域。\n可用于检索图谱的关键词：{'、'.join(f.keywords) or '无'}", evidence
