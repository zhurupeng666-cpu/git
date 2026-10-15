import time
import logging
from src.api.stream_control import check_cancelled
import os
import re
from dotenv import load_dotenv

from src.rag.v4_rerank import V4Rerank
from src.rag.answer_generator import AnswerGenerator
from src.rag.citation_verifier import CitationVerifier
from src.rag.answer_coverage_checker import AnswerCoverageChecker
from src.rag.engineer_queue import EngineerQueue

load_dotenv()

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG主流程：检索、生成、引用审核、覆盖度审核和工程师接管。"""

    def __init__(self):
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("未找到DATABASE_URL环境变量")

        # 1. 初始化V4检索器
        self.retriever = V4Rerank(db_url)

        # 2. 初始化答案生成器
        self.generator = AnswerGenerator()

        # 3. 两个审核器复用同一个DeepSeek客户端
        self.verifier = CitationVerifier(self.generator.llm)
        self.coverage_checker = AnswerCoverageChecker(self.generator.llm)

        # 4. 初始化工程师待答队列
        self.engineer_queue = EngineerQueue(db_url)

    def _is_approved(
            self,
            answer: str,
            verification: dict,
            coverage: dict,
            context=None
    ) -> bool:
        if verification.get("status") != "completed":
            return False

        if coverage.get("coverage_status") != "complete":
            return False

        if verification.get("invalid_references"):
            return False

        if verification.get("missing_aspects"):
            return False

        checks = verification.get("checks", [])
        if not isinstance(checks, list) or not checks:
            return False

        answer_refs = set(
            re.findall(r"\[\d+\]", answer)
        )
        if not answer_refs:
            return False

        checked_refs = set()

        for item in checks:

            if not isinstance(item, dict):
                return False

            if item.get("status") != "supported":
                return False

            refs = item.get("references")

            if not isinstance(refs, list) or not refs:
                return False

            checked_refs.update(
                [
                    f"[{ref}]"
                    for ref in refs
                ]
            )

        return answer_refs.issubset(
            checked_refs
        )

    @staticmethod
    def _get_supported_claims(answer: str, verification: dict, sources: list) -> list:
        """
        提取引用审核通过的技术结论。
        :param answer: 原始答案
        :param verification: 引用审核结果
        :param sources: 原始答案引用的证据来源
        :return: 有有效引用支持的技术结论列表
        """

        if verification.get("status") not in ("completed", "incomplete"):
            return []

        valid_refs = {item["reference"] for item in sources}
        original_refs = set(re.findall(r"\[\d+\]", answer))
        supported = []
        seen = set()

        for item in verification.get("checks", []):
            if not isinstance(item, dict) or item.get("status") != "supported":
                continue

            claim = item.get("claim")
            refs = item.get("references")

            if not isinstance(claim, str) or not claim.strip():
                continue

            if not isinstance(refs, list) or not refs:
                continue

            # 必须是原始答案使用过的、且真实存在的引用
            if not all(
                isinstance(ref, str)
                and ref in valid_refs
                and ref in original_refs
                for ref in refs
            ):
                continue

            claim = re.sub(r"\s*\[\d+\]", "", claim).strip()
            key = (claim, tuple(refs))

            if claim and key not in seen:
                supported.append({"claim": claim, "references": refs})
                seen.add(key)

        return supported

    @staticmethod
    def _get_unresolved_aspects(verification: dict, coverage: dict) -> list[str]:
        """
        整理需要工程师进一步处理的问题。
        :param verification: 引用审核结果
        :param coverage: 答案覆盖度审核结果
        :return: 未解决问题的描述列表
        """
        unresolved = []
        # 1. 优先使用覆盖度审核拆分出的用户子问题
        for item in coverage.get("aspects", []):
            if not isinstance(item, dict):
                continue

            status = item.get("status")
            aspect = item.get("aspect", "")

            if not isinstance(aspect, str) or not aspect.strip():
                continue

            if status == "partial":
                unresolved.append(f"尚未完整回答：{aspect.strip()}")

            elif status == "missing":
                unresolved.append(f"无法根据当前证据回答：{aspect.strip()}")

        # 2. 覆盖度审核没有识别出缺失项时，参考引用审核结果
        if not unresolved:
            for aspect in verification.get("missing_aspects", []):
                if isinstance(aspect, str) and aspect.strip():
                    unresolved.append(f"需要补充核实：{aspect.strip()}")

        # 3. 引用审核异常
        if verification.get("status") in ("error", "no_evidence"):
            unresolved.append("引用审核失败或缺少检索证据")

        # 4. 覆盖度审核异常
        if coverage.get("coverage_status") == "error":
            unresolved.append("问题覆盖度审核失败")

        # 5. 引用编号不存在
        if verification.get("invalid_references"):
            unresolved.append("原始答案存在无效引用，需要核实")

        # 6. 只有引用结论存在问题，但覆盖度没有发现缺失项
        if not unresolved:
            checks = verification.get("checks", [])

            if any(
                    item.get("status") in ("partial", "unsupported")
                    for item in checks
                    if isinstance(item, dict)
            ):
                unresolved.append("部分技术结论缺乏充分引用依据，需要工程师核实")

        # 7. 保底处理
        if not unresolved:
            unresolved.append("当前答案未通过完整审核，需要工程师复核")

        return list(dict.fromkeys(unresolved))


    def ask(self, query: str, top_k: int = 5, context = None) -> dict:
        """
        运行RAG，根据两个审核结果返回完整或部分答案。
        :param query: 用户问题
        :param top_k: 检索结果数量
        :return: 包含答案、来源、审核结果等信息的字典
        """

        if not query or not query.strip():
            raise ValueError("用户问题不能为空")

        if top_k < 1:
            raise ValueError("top_k必须大于等于1")

        # 第一步：V4检索 + 耗时监控
        from src.monitoring.rag_monitor import create_request_id, track_stage

        # 每次用户提问生成唯一请求ID
        request_id = create_request_id()

        # 自动记录检索耗时、执行状态与异常类型
        with track_stage(request_id, "retrieval"):
            retrieval_result = self.retriever.search(
                query=query,
                context=context,
                top_k=top_k
            )

        top_documents = retrieval_result["top_documents"]



        # 第二步：生成原始答案 + 耗时监控
        with track_stage(request_id, "generation"):
            generation_result = self.generator.generate(
                query=query,
                context=context,
                documents=top_documents
            )

        original_answer = generation_result["answer"]
        all_sources = generation_result["sources"]


        # 第三步：引用真实性审核 + 耗时监控
        try:
            with track_stage(request_id, "citation_verification"):
                verification = self.verifier.verify(
                    query=query,
                    context=context,
                    answer=original_answer,
                    documents=top_documents
                )

        except Exception as exc:
            verification = {
                "status": "error",
                "checks": [],
                "invalid_references": [],
                "missing_aspects": [],
                "error_type": type(exc).__name__
            }

        # 第四步：问题覆盖度检查

        # 第四步：问题覆盖度检查 + 耗时监控
        try:
            with track_stage(request_id, "coverage_initial"):
                coverage = self.coverage_checker.check(
                    query=query,
                    context=context,
                    answer=original_answer,
                    documents=top_documents
                )

        except Exception as exc:
            coverage = {
                "coverage_status": "error",
                "covered_aspects": [],
                "partial_aspects": [],
                "missing_aspects": [],
                "aspects": [],
                "error_type": type(exc).__name__
            }

        # 第五步：综合审核，判断是否完整通过
        is_approved = self._is_approved(
            original_answer,
            verification,
            coverage,
            context,
        )

        if is_approved:
            return {
                "query": query,
                "answer": original_answer,
                "sources": all_sources,
                "verified": True,
                "coverage_status": "complete",
                "coverage": coverage,
                "verification": verification,
                "handoff": None,
                "candidate_count": retrieval_result["candidate_count"],
                "top_chunk_ids": [doc["chunk_id"] for doc in top_documents]
            }

        # 第六步：提取有依据的技术结论
        supported = self._get_supported_claims(
            original_answer,
            verification,
            all_sources
        )

        if coverage.get("coverage_status") not in ("complete", "partial"):
            supported = []

        # 根据审核通过的结论构建最终可展示答案
        if supported:
            filtered_answer = "\n".join(
                f"{i}. {item['claim']} {' '.join(item['references'])}"
                for i, item in enumerate(supported, start=1)
            )
        else:
            filtered_answer = "当前没有已通过引用审核的技术结论。"

        # 重新检查最终答案，而不是继续使用原始答案的覆盖度

        # 重新检查最终答案 + 耗时监控
        try:
            with track_stage(request_id, "coverage_final"):
                final_coverage = self.coverage_checker.check(
                    query=query,
                    context=context,
                    answer=filtered_answer,
                    documents=top_documents
                )

        except Exception as exc:
            final_coverage = {
                "coverage_status": "error",
                "covered_aspects": [],
                "partial_aspects": [],
                "missing_aspects": [],
                "aspects": [],
                "error_type": type(exc).__name__
            }

        # 使用最终答案的覆盖度决定哪些问题需要工程师补充
        coverage = final_coverage
        unresolved = self._get_unresolved_aspects(verification, coverage)

        # 如果存在尚未审核到的原始引用，增加核实说明
        answer_refs = set(re.findall(r"\[\d+\]", original_answer))
        checked_refs = {
            ref
            for item in verification.get("checks", [])
            if isinstance(item, dict)
            for ref in item.get("references", [])
        }

        if answer_refs - checked_refs:
            unresolved.append("原始答案中的部分引用未完成审核")

        unresolved = list(dict.fromkeys(unresolved))

        # 仅保留已支持结论对应的来源
        supported_refs = {
            ref
            for item in supported
            for ref in item["references"]
        }

        supported_sources = [
            source for source in all_sources
            if source["reference"] in supported_refs
        ]

        # 第七步：组织用户可见的部分答案
        answer_parts = []

        if supported:
            lines = [
                f"{index}. {item['claim']} {' '.join(item['references'])}"
                for index, item in enumerate(supported, start=1)
            ]

            answer_parts.append(
                "根据当前检索到的知识库证据，可以确认以下内容：\n"
                + "\n".join(lines)
            )
        else:
            answer_parts.append(
                "当前没有足够的、已通过自动证据审核的内容可供直接回答。"
            )

        answer_parts.append(
            "以下问题目前无法根据已检索的资料充分确认：\n"
            + "\n".join(f"- {aspect}" for aspect in unresolved)
        )

        # 第八步：仅将未解决部分作为工程师工单重点
        engineer_question = (
            f"用户原始问题：{query}\n\n"
            "需要工程师补充或核实的部分：\n"
            + "\n".join(f"- {aspect}" for aspect in unresolved)
        )


        # 第八步：工程师工单提交 + 耗时监控
        try:
            with track_stage(request_id, "engineer_handoff"):

                handoff = self.engineer_queue.submit(
                    question=engineer_question,
                    verification={
                        **verification,
                        "coverage": coverage,
                        "unresolved_aspects": unresolved
                    },
                    evidence_refs=all_sources
                )

                if not isinstance(handoff, dict) or not handoff.get("question_id"):
                    raise ValueError("工程师工单未返回有效的问题编号")


            answer_parts.append(
                f"未解决的问题已提交工程师。\n"
                f"问题编号：{handoff['question_id']}\n"
                "当前状态：等待工程师回答。"
            )

        except Exception as exc:
            handoff = {
                "status": "failed",
                "error_type": type(exc).__name__
            }

            answer_parts.append(
                "未解决的问题暂时未能成功提交工程师，请稍后重试。"
            )

        # 第九步：返回部分答案 + 覆盖情况 + 转交状态
        return {
            "query": query,
            "answer": "\n\n".join(answer_parts),
            "sources": supported_sources,
            "verified": False,
            "coverage_status": coverage.get("coverage_status"),
            "coverage": coverage,
            "verification": verification,
            "unresolved_aspects": unresolved,
            "handoff": handoff,
            "candidate_count": retrieval_result["candidate_count"],
            "top_chunk_ids": [doc["chunk_id"] for doc in top_documents]
        }

    def ask_stream(self, query: str, top_k=5, context=None):
        """检索完成后立即转发生成器 token，保留空白和换行。"""
        if not query or not query.strip():
            raise ValueError("用户问题不能为空")
        if top_k < 1:
            raise ValueError("top_k必须大于等于1")

        started = time.monotonic()
        logger.info("RAG retrieval started")
        check_cancelled()
        retrieval_result = self.retriever.search(
            query=query, context=context, top_k=top_k
        )
        check_cancelled()
        top_documents = retrieval_result["top_documents"]
        logger.info("RAG retrieval completed in %.2fs; documents=%d",
                    time.monotonic() - started, len(top_documents))

        yield {"type": "status", "data": {
            "stage": "generation", "message": "正在生成工程分析结果..."
        }}
        sources = []
        for index, doc in enumerate(top_documents, start=1):
            metadata = doc.get("metadata") or {}
            sources.append({
                "reference": f"[{index}]",
                "file_name": metadata.get("file_name"),
                "domain": metadata.get("domain"),
                "title": metadata.get("section_title") or metadata.get("title"),
                "page": metadata.get("page_number"),
                "chunk_id": doc.get("chunk_id", metadata.get("chunk_id"))
            })
        # 来源可在等待首个模型 token 时显示。
        yield {"type": "source", "sources": sources}

        started = time.monotonic()
        first_token = True
        stream = self.generator.stream_generate(
            query=query, documents=top_documents, context=context
        )
        try:
            for chunk in stream:
                check_cancelled()
                if chunk:
                    if first_token:
                        logger.info("RAG first token received in %.2fs",
                                    time.monotonic() - started)
                        first_token = False
                    yield {"type": "chunk", "content": chunk}
            if first_token:
                raise RuntimeError("模型流已结束，但没有返回答案文本")
            logger.info("RAG generation completed in %.2fs", time.monotonic() - started)
        finally:
            stream.close()
