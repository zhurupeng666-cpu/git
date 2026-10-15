"""优先使用本机完整模型快照，避免每个Worker启动时重复联网检查。"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def model_path(model_name: str) -> str:
    if Path(model_name).is_dir():
        return model_name
    # 默认离线读取已下载权重；首次部署显式开启下载，失败时保留原始异常。
    allow_download = os.getenv("LOCAL_MODEL_ALLOW_DOWNLOAD", "0") == "1"
    try:
        return snapshot_download(model_name, local_files_only=not allow_download)
    except Exception as exc:
        raise RuntimeError(
            f"无法读取本地模型{model_name}，请先下载完整权重；"
            "首次部署可设置LOCAL_MODEL_ALLOW_DOWNLOAD=1。"
        ) from exc
