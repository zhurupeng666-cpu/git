"""主后端仅调用隔离的OCR进程，不导入Paddle或修改PyTorch运行环境。"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
from threading import Lock
from uuid import uuid4
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
OCR_LOCK = Lock()


class OcrEngine:
    def __init__(self, python_path=None, dpi=None):
        self.python = Path(
            python_path or os.getenv("OCR_PYTHON", str(ROOT / ".ocr-venv/Scripts/python.exe"))
        )
        self.dpi = int(dpi or os.getenv("OCR_RENDER_DPI", "180"))
        self.orientation = os.getenv("OCR_CORRECT_ORIENTATION", "1") == "1"
        if not 100 <= self.dpi <= 300:
            raise ValueError("OCR_RENDER_DPI应在100到300之间")
        self.folder = Path(os.getenv("OCR_CACHE_DIR", str(ROOT / "data/ocr_cache")))
        self.folder.mkdir(parents=True, exist_ok=True)

    def parse(self, source, pages, source_sha256=None):
        if not self.python.is_file():
            raise RuntimeError("缺少独立OCR环境，请先运行scripts/setup_ocr.ps1")
        source = Path(source).resolve()
        digest = source_sha256 or hashlib.sha256(source.read_bytes()).hexdigest()
        job_id = uuid4().hex
        request, result, log = [
            self.folder / (job_id + suffix) for suffix in (".request.json", ".result.json", ".log")
        ]
        request.write_text(
            json.dumps(
                {
                    "source": str(source),
                    "source_sha256": digest,
                    "pages": list(pages),
                    "dpi": self.dpi,
                    "orientation": self.orientation,
                    "cache_dir": str(self.folder / "pages"),
                }
            ),
            encoding="utf-8",
        )
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        proxy = os.getenv("OCR_MODEL_DOWNLOAD_PROXY")
        if proxy:
            env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy)
        timeout = float(os.getenv("OCR_PROCESS_TIMEOUT_SECONDS", "1800"))
        try:
            # 每次只运行一个文件，避免多个Paddle进程争用内存；超时由subprocess终止并等待回收。
            with OCR_LOCK, FileLock(str(self.folder / "worker.lock"), timeout=timeout), log.open(
                "w", encoding="utf-8"
            ) as stream:
                process = subprocess.run(
                    [
                        str(self.python),
                        str(ROOT / "scripts/ocr_worker.py"),
                        "--request",
                        str(request),
                        "--result",
                        str(result),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
            if process.returncode or not result.exists():
                raise RuntimeError(f"OCR解析失败，请查看本地日志：{log}")
            return json.loads(result.read_text(encoding="utf-8"))["pages"]
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"OCR工作进程超时，已终止；日志：{log}") from exc
