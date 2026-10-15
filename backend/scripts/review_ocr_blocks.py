"""根据逐块决定生成新的复核预览，原始识别结果不覆盖，不直接发布数据库。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ingestion.ocr_review import review_preview

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(review_preview(args.preview, args.decisions, args.output_dir))
