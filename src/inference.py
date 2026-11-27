"""共享的单图解码、checkpoint 校验与推理入口。"""

import hashlib
import io
import warnings
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from src.data.dataset import build_transforms, load_project_config
from src.models.c0 import C0ResNet18

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CLASS_NAMES = {"Cr": "龟裂", "In": "夹杂", "Pa": "斑块", "PS": "麻点", "RS": "氧化铁皮压入", "Sc": "划痕"}


def decode_upload(content: bytes, filename: str) -> Image.Image:
    if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("请上传 JPEG 或 PNG 图片。")
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("图片不能为空，大小须在 10 MB 以内。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("文件内容须为 JPEG 或 PNG 图片。")
                image.load()
                return image.convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("图片损坏或像素规模过大，请重新选择图片。") from error


def reset_confirmation(state, identity):
    """绑定图片和权重内容，清除上一输入的推理及确认状态。"""
    if state.get("input_identity") != identity:
        for key in ("prediction", "confirmed_class", "class_choice", "conditions_ok"):
            state.pop(key, None)
        state["input_identity"] = identity


class Predictor:
    def __init__(self, root: Path, checkpoint: Path, device: str = "auto"):
        self.root = root
        config, self.class_map = load_project_config(root)
        if set(self.class_map) != set(CLASS_NAMES) or sorted(self.class_map.values()) != list(range(6)):
            raise ValueError("类别映射必须包含六类和 0 到 5 的唯一编号。")
        self.names = sorted(self.class_map, key=self.class_map.get)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        if metadata["class_map"] != self.class_map or metadata["preprocessing"] != config["preprocessing"]:
            raise ValueError("权重中的类别映射或预处理与项目配置不一致。")
        self.model = C0ResNet18(pretrained=False)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.transform = build_transforms("val", config)
        self.model_id = metadata["experiment_id"]
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0].cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError("模型输出异常，请检查输入和权重。")
        predicted = self.names[int(probabilities.argmax())]
        return {
            "model_id": self.model_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": str(self.device),
            "predicted_class": predicted,
            "predicted_name": CLASS_NAMES[predicted],
            "scores": dict(zip(self.names, probabilities.tolist())),
        }
