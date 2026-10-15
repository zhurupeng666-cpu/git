"""独立Python 3.11 OCR工作进程；逐页渲染，不向外部大模型发送文档。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / "data/ocr_models"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from src.ingestion.document_blocks import parse_table_html, table_markdown, restore_rotated_bbox


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def normalized_box(box, width, height):
    values = [float(value) for value in box]
    return [max(0, min(1, values[i] / (width if i % 2 == 0 else height))) for i in range(4)]


def normalize_result(result, page_number, width, height):
    # JSON接口把numpy数组转换为列表，避免把SDK结果类型泄漏到主后端。
    data = result.json
    if isinstance(data, str):
        data = json.loads(data)
    data = data.get("res", data)
    angle = int(data.get("doc_preprocessor_res", {}).get("angle", 0) or 0)
    if angle < 0:
        angle = 0
    processed_width, processed_height = (height, width) if angle in (90, 270) else (width, height)
    ocr = data.get("overall_ocr_res", {})
    texts, scores = ocr.get("rec_texts", []), ocr.get("rec_scores", [])
    boxes = ocr.get("rec_boxes", [])
    if not len(boxes):
        boxes = []
        for polygon in ocr.get("rec_polys", []):
            boxes.append(
                [
                    min(point[0] for point in polygon),
                    min(point[1] for point in polygon),
                    max(point[0] for point in polygon),
                    max(point[1] for point in polygon),
                ]
            )
    lines = [
        {"text": text, "confidence": float(score), "bbox": list(box)}
        for text, score, box in zip(texts, scores, boxes)
    ]
    blocks = []
    skipped = []
    for index, item in enumerate(data.get("parsing_res_list", [])):
        kind = item.get("block_label", "text")
        if kind in ("header", "footer", "page_number"):
            skipped.append({"kind": kind, "text": item.get("block_content", "")})
            continue
        box = list(item["block_bbox"])
        selected = [
            line
            for line in lines
            if box[0] <= (line["bbox"][0] + line["bbox"][2]) / 2 <= box[2]
            and box[1] <= (line["bbox"][1] + line["bbox"][3]) / 2 <= box[3]
        ]
        content = item.get("block_content", "") or ""
        details = {
            "lines": [
                {"text": line["text"], "confidence": line["confidence"]} for line in selected
            ],
            "orientation_angle": angle,
        }
        if kind == "table":
            table_matches = [
                value
                for value in data.get("table_res_list", [])
                if value.get("pred_html") == content
            ]
            if table_matches:
                table_ocr = table_matches[0].get("table_ocr_pred", {})
                table_lines = [
                    {"text": text, "confidence": float(score)}
                    for text, score in zip(
                        table_ocr.get("rec_texts", []), table_ocr.get("rec_scores", [])
                    )
                ]
                if table_lines:
                    selected = table_lines
                    details["lines"] = table_lines
            try:
                rows = parse_table_html(content)
            except (ValueError, TypeError):
                rows = []
                details["table_parse_error"] = "invalid_table_structure"
            details["table_rows"] = rows
            details["raw_table_html"] = content
            content = table_markdown(rows)
        if kind in ("image", "figure", "chart"):
            # OCR只记录图中文字；绝不把箭头/连线关系当作已经由视觉模型理解。
            content = "\n".join(line["text"] for line in selected)
            details["visual_semantics_verified"] = False
        confidence = (
            sum(line["confidence"] for line in selected) / len(selected) if selected else None
        )
        blocks.append(
            {
                "page_number": page_number,
                "block_index": index,
                "kind": kind,
                "text": content,
                "bbox": restore_rotated_bbox(
                    normalized_box(box, processed_width, processed_height), angle
                ),
                "method": "paddle_structure_v3",
                "confidence": confidence,
                "details": details,
            }
        )
    if not blocks:
        raise RuntimeError("OCR未返回版面块，不能按空页成功处理")
    return {
        "page_number": page_number,
        "width": width,
        "height": height,
        "blocks": blocks,
        "excluded_margins": skipped,
        "engine": "PaddleOCR-3.3.2/PP-StructureV3",
        "ocr_text_lines": len(lines),
    }


def build_pipeline(orientation):
    from paddleocr import PPStructureV3
    from paddlex.inference.pipelines.layout_parsing.pipeline_v2 import _LayoutParsingPipelineV2

    original_factory = _LayoutParsingPipelineV2.create_model

    def create_enabled_model(pipeline, config, *args, **kwargs):
        # 固定版本PaddleX 3.3.x会无条件初始化Chart2Table；只跳过明确禁用的图表模块。
        # 修补仅发生在本次隔离进程构造期间，不修改第三方安装文件或后端环境。
        if config.get("model_name") == "PP-Chart2Table" and not pipeline.use_chart_recognition:
            return None
        return original_factory(pipeline, config, *args, **kwargs)

    _LayoutParsingPipelineV2.create_model = create_enabled_model
    try:
        return PPStructureV3(
            device="cpu",
            cpu_threads=2,
            enable_mkldnn=False,
            use_doc_orientation_classify=orientation,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_formula_recognition=False,
            use_seal_recognition=False,
            use_chart_recognition=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
        )
    finally:
        _LayoutParsingPipelineV2.create_model = original_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    source = Path(request["source"])
    cache = Path(request["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    pipeline = None
    pdf = None
    responses = []
    import numpy as np
    from PIL import Image, ImageOps

    if source.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(source)
    try:
        for page_number in request["pages"]:
            signature = f"{request['source_sha256']}|{page_number}|{request['dpi']}|{request.get('orientation',True)}|structure-v2"
            cached = cache / (hashlib.sha256(signature.encode()).hexdigest() + ".json")
            if cached.exists():
                responses.append(json.loads(cached.read_text(encoding="utf-8")))
                continue
            page = bitmap = None
            if pdf:
                page = pdf[page_number - 1]
                if page.get_width() * page.get_height() * (request["dpi"] / 72) ** 2 > 25000000:
                    page.close()
                    raise ValueError("PDF页面渲染后超过2500万像素，请先拆分或缩小页面")
                bitmap = page.render(scale=request["dpi"] / 72)
                image = bitmap.to_pil().convert("RGB")
            else:
                with Image.open(source) as original:
                    if original.width * original.height > 25000000:
                        raise ValueError("图片超过2500万像素，请先按区域拆分")
                    if getattr(original, "n_frames", 1) > 1:
                        raise ValueError("多帧图片请先转换为逐页PDF，不能只识别第一帧")
                    corrected = ImageOps.exif_transpose(original)
                    if corrected.mode in ("RGBA", "LA") or "transparency" in corrected.info:
                        rgba = corrected.convert("RGBA")
                        image = Image.new("RGB", rgba.size, "white")
                        image.paste(rgba, mask=rgba.getchannel("A"))
                        rgba.close()
                    else:
                        image = corrected.convert("RGB")
            try:
                if image.width * image.height > 25000000:
                    raise ValueError("单页图片超过2500万像素，请先按页或区域拆分")
                if pipeline is None:
                    pipeline = build_pipeline(request.get("orientation", True))
                output = next(iter(pipeline.predict(np.asarray(image))))
                value = normalize_result(output, page_number, image.width, image.height)
                atomic_json(cached, value)
                responses.append(value)
                print(
                    f"OCR page {page_number} completed; blocks={len(value['blocks'])}", flush=True
                )
            finally:
                image.close()
                if bitmap:
                    bitmap.close()
                if page:
                    page.close()
    finally:
        if pdf:
            pdf.close()
    atomic_json(Path(args.result), {"pages": responses})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
