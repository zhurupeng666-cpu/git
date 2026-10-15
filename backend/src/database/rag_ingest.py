"""兼容旧入口：默认只预览；正式发布必须显式传入--publish，并先备份知识库。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from scripts.ingest_documents import main

    main()
