# 图片和扫描PDF入库

当前链路保持单Agent：文档在离线导入阶段由独立Python 3.11进程识别，聊天时继续使用原来的BGE-M3、BM25、扩展问题向量和重排。

## 环境

- 后端原`.venv`保持不变，不安装Paddle。
- `.ocr-venv`优先安装锁定的`requirements-ocr.lock.txt`，缺失时使用`requirements-ocr.txt`。识别引擎为PaddleOCR 3.3.2 / PP-StructureV3，CPU版Paddle 3.2.2。
- 默认CPU推理，两个线程；OCR进程按文件串行运行，主进程还通过文件锁防止多个导入命令同时占用模型。
- 模型在`data/ocr_models`，逐页解析缓存位于`data/ocr_cache`，预览及数据库备份位于`data/ocr_runs`。这些目录不应提交公开仓库。
- 首次下载可通过`OCR_MODEL_DOWNLOAD_PROXY`为工作进程单独指定代理；不会改变主后端或全局系统代理。

```powershell
.\scripts\setup_ocr.ps1
```

如需要代理：

```powershell
.\scripts\setup_ocr.ps1 -DownloadProxy http://127.0.0.1:7897
$env:OCR_MODEL_DOWNLOAD_PROXY = 'http://127.0.0.1:7897'
```

## 先预览

```powershell
.\.venv\Scripts\python.exe scripts/ingest_documents.py --new-library
```

`--new-library`只处理公开知识来源清单中的四册扫描PDF；配套图片关联到PDF，不作为独立文档重复发布。普通独立图片可以用`--file`指定。

`--max-pages 2`可以只解析前两页，但截断预览禁止发布。默认不传`--publish`时不会创建数据库表或写入知识库。

每次预览生成：

- `manifest.json`：来源校验值、页数、页面类型、切片ID及待复核项。
- `raw_blocks.json`：原始块内容、位置、表格和识别置信度。
- `chunks.jsonl`：通过识别质量门禁的待发布切片。

OCR平均置信度不代表事实准确率。含低置信度数字／型号、表格结构缺失、未知视觉关系的区域不会自动放入正式检索。图中箭头、曲线和复杂结构仍需视觉理解或人工复核；本链路不冒充已经支持完整多模态推理。

原生PDF继续读取文字；扫描和大图混合页采用整页版面解析，不把隐藏文字层重复叠加。表格分片重复表头，保留原始HTML和区域坐标。DOCX继续采用现有标题解析，不伪造Word页码。

方向校正支持直角旋转，引用区域回映到校正前原页；不进行自由形变展开。单页渲染前检查像素预算，单文件限制100MB、1000页，超大资料需要先拆分。

## 逐块复核

如需校正待复核项，先查看原始页面，再准备JSON决定文件；不能仅因模型置信度高就声明事实正确。例如：

```json
{
  "reviewer": "实际复核者标识",
  "decisions": [{
    "document_id": "从预览清单复制",
    "source_sha256": "从预览清单复制",
    "page_number": 1,
    "block_index": 3,
    "action": "replace_text",
    "text": "对照原图确认的完整文字",
    "reason": "说明原图依据和校正内容"
  }]
}
```

运行：

```powershell
.\.venv\Scripts\python.exe scripts/review_ocr_blocks.py --preview '原预览目录' --decisions '决定文件.json' --output-dir '新复核预览目录'
```

支持`replace_text`、`replace_table`（提供`rows`行列数组）、`describe_figure`、`confirm_text`和`exclude`。工具保留原预览，输出校正审计，并重新执行切片和门禁；不直接写数据库。复核者标识会如实记录，AI辅助转写复核不冒充企业工程师审核。

## 发布

确认预览后执行：

```powershell
.\.venv\Scripts\python.exe scripts/ingest_documents.py --publish --from-preview '上一步输出的预览目录'
```

发布前通过`pg_dump`备份三张知识表；已存在的OCR版本表也会一并备份。密码通过进程环境变量传递，不出现在命令参数或输出中。备份格式会由`pg_restore --list`检查。找不到`pg_dump`时可以设置`PG_DUMP_PATH`，未完成备份不会发布。

发布前生成BGE正文向量、摘要、五个问题及对应向量。辅助问题基于切片调用现有DeepSeek接口，可能产生费用；OCR本身只在本机运行。所有正文、辅助索引和文档版本在同一个PG事务里发布，发生异常整体回滚。

同一文件内容、解析配置未变化时不重复发布。更新只替换当前来源的切片，不按文件名模糊删除其他文档。已有文档的新版本仍有待复核项时会保留旧版本；首次导入时只发布通过门禁的块，待复核区域保留在报告中，并记录未发布数量。

BM25按默认5秒间隔检查数据库中的发布标记，发现变化后替换不可变快照；正文向量查询直接读取PG。进程内缓存不是跨服务的原子快照，运行中的请求可能处于发布边界，短暂刷新延迟是明确的实现边界。

旧入口`python -m src.database.rag_ingest`委托同一命令行流程，默认也只预览，避免绕过备份直接写库。

## 引用与检查

后端来源保留页码、区域序号、归一化坐标及解析方式；前端展示原文页码和区域。暂未增加点击高亮原图的浏览接口。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/test_ocr_publication_pg.py
```

第二项在真实PG创建临时隔离schema，模拟中途失败、重复发布和辅助向量写入；完成后清理，并核对正式知识库未被测试修改。它不调用真实模型，也不等于完整问答质量验收。

OCR配置：`OCR_PYTHON`、`OCR_RENDER_DPI`（100-300，默认180）、`OCR_MIN_CONFIDENCE`（默认0.85）、`OCR_CACHE_DIR`、`OCR_PROCESS_TIMEOUT_SECONDS`（默认1800）、`KNOWLEDGE_INDEX_POLL_SECONDS`（默认5）。修改门禁配置后重新预览，不要直接手工将待复核项标为通过。

`OCR_CORRECT_ORIENTATION`默认1。固定PaddleX 3.3.13会初始化已禁用的图表模型，工作进程在构造期间显式跳过该模块；不会修改第三方安装文件，也不据此宣称支持图表语义识别。
