# OCR接入与验收记录

## 实现范围

保持现有单Agent聊天架构。OCR只在离线文档导入时启动独立Python 3.11工作进程，原后端Python 3.14环境不安装Paddle。原生PDF继续提取原文，扫描和混合页使用PaddleOCR 3.3.2 / PP-StructureV3。独立图片支持PNG、JPEG、TIFF等格式；PDF与配套图片关联去重。

| 文件或目录 | 改动目的 |
| --- | --- |
| `src/ingestion/pdf_page_parser.py`、`image_parser.py`、`ocr_engine.py`、`scripts/ocr_worker.py` | 页面分类、隔离OCR、旋转定位和逐页缓存 |
| `src/ingestion/document_blocks.py`、`ocr_quality_checker.py`、`structured_chunker.py` | 保留版面区域、表格行列、质量门禁及Token预算切片 |
| `src/ingestion/ocr_review.py`、`scripts/review_ocr_blocks.py` | 逐块校正、原始结果保留和复核审计 |
| `src/ingestion/ingestion_service.py`、`ingestion_publisher.py`、`scripts/ingest_documents.py` | 预览、真实PG备份、BGE向量和辅助索引事务发布 |
| `scripts/ocr_ingestion_migration.sql` | 文档来源版本、哈希及待复核数量记录 |
| `src/ingestion/pdf_chunker.py`、`knowledge_base_chunker.py`、`document_loader.py`、`src/database/rag_ingest.py` | 接入原有加载入口，原生资料原切分方式保留 |
| `src/rag/index_snapshot.py`、`bm25_retriever.py`、`keyword_bm25_retriever.py` | 发布后自动发现知识变化并刷新BM25，查询使用一致快照 |
| `src/rag/answer_generator.py`、`rag_pipeline.py` | 向回答及来源事件传递页码、区域和资料属性 |
| 前端 `src/api/sessions.ts`、`src/components/ChatBox.vue` | 来源展示页码和区域 |
| `requirements-ocr.txt`、`requirements-ocr.lock.txt`、`scripts/setup_ocr.ps1` | 独立且可复现的OCR依赖安装 |

## 已完成的代码验证

- 完整自动回归：76项通过，其中24项覆盖OCR新增逻辑；原始运行日志已在项目整理时清除。
- 前端类型检查与生产构建通过；没有自动重启现有业务服务。
- 真实PostgreSQL隔离schema检查通过：故障注入后事务回滚、重复发布不增加记录、每片一条摘要和五条问题向量、原库不变、测试schema清理。
- 实际扫描PDF与独立PNG均完成OCR。配套PNG只验证识别入口，不与其PDF重复入库。

## 资料与复核

四册资料共36页，均实际经过OCR，未把编写原文或来源清单当作识别结果。最终预览：`data/ocr_runs/public_library_accepted`。

识别初始产生54个可用切片、21个待复核区域。对照原图校正了3处正文识别错误和2处表格字形／符号转写差异，复核者标记为“AI辅助原图转写复核”，不冒充企业工程师审核。校正前识别块保留为`original_ocr_blocks.json`，校正理由和原文校验值见`review_audit.json`。本批18张表的行列文字已与编写底稿比对通过（仅规范空白和全角字符后比较）；这项验收不把底稿作为OCR输入，也不代表其他资料均有同样识别准确率。余下18个图示区域只保留在复核报告，不进入正式向量检索。

本批资料是AI辅助编写的公开知识综述，不是企业受控SOP；该属性也进入检索上下文。OCR置信度仅代表识别指标，不代表专业事实审核。

## 实际数据库发布与问答

已于2026-10-15发布到项目实际配置的PostgreSQL，使用真实BGE-M3编码和现有DeepSeek辅助索引生成，没有向正式库写测试向量。

| 表 | 原有记录 | 新增记录 | 发布后 |
| --- | ---: | ---: | ---: |
| rag_chunks | 877 | 54 | 931 |
| rag_chunk_expansions | 877 | 54 | 931 |
| rag_chunk_expansion_vectors | 5262 | 324 | 5586 |

每个新增切片包含1份摘要、5个扩展问题及6条扩展向量；同时记录4个文档版本。原有三表所有行的逐行哈希比对通过；新增切片的质量状态、页码和区域定位均完整。再次发布同一预览返回`already_published`，没有新增记录。

发布前备份：`data/ocr_runs/public_library_accepted/knowledge_before_publish_a4fc32ac0878.dump`（31,469,400字节）。SHA-256：`0e680348840ea940ce142bb1517467e3382693b1679df95779bc5e7fca90bf2a`。通过pg_restore目录校验。`.env`仅新增已验证的本机`PG_DUMP_PATH`工具路径，未修改数据库凭证。

逐行完整性验收和18张表格转写比对已通过；临时JSON验收输出已清理，结论保留在本文。

真实抽样问答完成：问题为“电容薄膜压力计对气体组分不敏感，是否就意味着适用于所有介质？”。

- BM25与正文向量检索的Top1均为新增扫描PDF设备分册第1页。
- 现有融合与Top20重排之后，前两条来源为新增扫描PDF第1、2页，均带归一化区域坐标和质量状态。
- 流式生成成功返回答案，核心回答区分了测量原理与介质兼容性，包含引用编号。本次问答临时JSON输出已清理。
- 这是一次实际检索生成连通性抽样，不是全量召回率评测或逐句事实审核。本次计时包含冷进程模型加载，不作为线上SSE性能基准。

## 使用边界

- 图示箭头、曲线、复杂结构不自动推断；需要逐块提交图示说明后重新复核发布。
- 当前引用可展示页码和区域，还没有点击后高亮原图的文件查看器。
- 100MB、1000页和2500万像素是单文件/页面资源边界；多帧图片请先转逐页PDF。
- 新增资料会改变全库排序，未开展全量召回率对比；现有BGE模型、融合权重、Top20重排、批次2和推理线程2设置保持原值。
- 运行中的旧后端需重启一次加载代码，之后新发布可由BM25轮询发现。没有启动额外后台服务。
- 数据库备份只覆盖知识库相关表，不包含用户、业务、登录会话表；已检查备份目录，但未执行生产库恢复演练。

## 运行资料整理

已清理试运行、重复预览、逐片扩展JSON缓存和临时日志。正式归档保留manifest.json、chunks.jsonl、raw_blocks.json、original_ocr_blocks.json、review_audit.json及发布前数据库备份。original_ocr_blocks.json为校正前的原始识别块；逐页OCR缓存已按要求清空，后续导入需要重新识别；模型权重保留以供复用。清理不删除PostgreSQL中的任何记录。
