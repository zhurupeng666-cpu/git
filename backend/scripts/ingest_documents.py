"""默认只解析预览；--publish才会备份并向真实PG发布质量通过的切片。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.ingestion.ingestion_service import (
    IMAGE_SUFFIXES,
    library_links,
    parse_document,
    save_preview,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "data/knowledge_base"))
    parser.add_argument("--file", action="append", help="可多次指定，路径必须在知识库根目录内")
    parser.add_argument("--new-library", action="store_true", help="只处理公开知识清单中的四册PDF")
    parser.add_argument("--max-pages", type=int, help="仅预览前N页，截断文档禁止发布")
    parser.add_argument("--output-dir")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--from-preview", help="发布已有完整预览，不重复OCR")
    args = parser.parse_args()
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages必须大于0")
    if args.from_preview:
        if not args.publish:
            parser.error("--from-preview需要同时指定--publish")
        folder = Path(args.from_preview)
    else:
        root = Path(args.root).resolve()
        linked, _ = library_links(root)
        skipped = []
        if args.file:
            paths = [Path(path).resolve() for path in args.file]
        elif args.new_library:
            paths = [root / path for path in sorted({item["pdf"] for item in linked.values()})]
        else:
            paths = []
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES | {".pdf", ".docx"}:
                    relative = path.relative_to(root).as_posix()
                    if relative in linked:
                        skipped.append(
                            {"source": relative, "reason": "linked_image_uses_canonical_pdf"}
                        )
                    else:
                        paths.append(path)
        if not paths:
            parser.error("没有找到待解析的PDF或图片")
        documents = []
        for path in paths:
            print(f"Parsing {path.name}", flush=True)
            documents.append(parse_document(path, root, max_pages=args.max_pages))
        folder = save_preview(documents, skipped, output_dir=args.output_dir)
        print(
            json.dumps(
                {
                    "preview": str(folder),
                    "documents": len(documents),
                    "chunks": sum(len(doc["chunks"]) for doc in documents),
                    "review_items": sum(len(doc["review_items"]) for doc in documents),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.publish:
        from src.ingestion.ingestion_publisher import publish_preview

        print(json.dumps(publish_preview(folder), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
