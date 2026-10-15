
import os

import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv

load_dotenv()


class EngineerQueue:
    """工程师待答队列：负责将AI无法可靠回答的问题保存到PostgreSQL。"""

    def __init__(self, db_url: str | None = None):
        # 复用RAG项目现有的PostgreSQL数据库
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("未找到DATABASE_URL环境变量")

    def submit(self, question: str, verification: dict, evidence_refs: list[dict]) -> dict:
        """提交待答问题，返回数据库生成的问题编号和状态。"""

        if not question or not question.strip():
            raise ValueError("提交给工程师的问题不能为空")

        # 只保存引用定位信息，不存储完整Chunk正文
        safe_refs = [
            {
                "reference": item.get("reference"),
                "chunk_id": item.get("chunk_id"),
                "file_name": item.get("file_name"),
                "location": item.get("location")
            }
            for item in evidence_refs
        ]

        sql = """
            INSERT INTO engineer_questions (question, verification, evidence_refs)
            VALUES (%s, %s, %s)
            RETURNING id, status, created_at
        """

        # 使用事务保证提交成功后才返回问题编号
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (
                    question.strip(),
                    Jsonb(verification),
                    Jsonb(safe_refs)
                ))
                question_id, status, created_at = cursor.fetchone()

        return {
            "question_id": question_id,
            "status": status,
            "created_at": created_at.isoformat()
        }
