# 半导体智能知识库

Vue + FastAPI + LangGraph 单 Agent 项目，结合 PostgreSQL / pgvector、BGE-M3 编码、BGE 重排与外部模型接口。

## 功能

- 半导体工艺、设备维护、故障排查与规范问答，支持 SSE 输出和来源展示。
- 当前库存、出入库流水、周期库存统计，以及按材料范围生成周报、月报。
- 工程师待处理队列、人工接单和回复。
- 登录会话、Tab Token、会话历史、短期上下文、增量摘要及长期记忆。
- 检索评测、自动回归与 Locust 压测脚本。

## 目录

```text
backend/                 后端、导入脚本和测试
  src/                   业务源码
  tests/                 自动回归
  scripts/               启动、Tokenizer准备、记忆测试和压测
  data/dictionaries/     专业分词词典
  .env.example           配置模板，不含真实凭证
  requirements.lock.txt 当前Windows开发与评测环境的完整依赖快照
frontend/                Vue前端及package-lock.json
```

知识库文档、结构化业务数据、评测语料及原文结果、数据库备份、真实配置、日志、模型权重和虚拟环境不随公开源码上传。原项目中的这些文件保持不变。

## 运行前提

当前后端在 Windows、Python 3.14 环境开发。依赖快照包含评测工具，并非最小生产依赖；GPU版PyTorch需按目标硬件另行安装。

前端Node版本要求以 `frontend/package.json` 为准：`^22.18.0 || >=24.12.0`。

需要自行准备：

1. PostgreSQL、pgvector，以及原项目所需业务表、认证表、工单表和知识库数据。此源码快照尚未包含一套可从空库完整初始化的数据库迁移脚本，不能只创建空数据库就运行。
2. DeepSeek与OpenRouter接口配置；模型名称可通过环境变量配置，具体可用性取决于接口账户。
3. `BAAI/bge-m3` 与 `BAAI/bge-reranker-v2-m3` 完整本地模型缓存。默认离线加载；首次准备模型可使用 `LOCAL_MODEL_ALLOW_DOWNLOAD=1`，完成后恢复为0。
4. 需要重新导入资料时，自行放入 `backend/data/knowledge_base/`。导入前先备份数据库，确认路径与已有索引版本，避免生成重复或残留内容。

## 本地启动（PowerShell）

后端：

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
Copy-Item .env.example .env
```

编辑 `.env`，填写数据库连接、接口密钥和至少32字符的随机JWT密钥。之后准备参考分词表并启动：

```powershell
.\.venv\Scripts\python.exe scripts/setup_context_tokenizer.py
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

在另一个终端启动前端：

```powershell
cd frontend
npm ci
npm run dev
```

访问 `http://localhost:5173`，前端开发代理将 `/api` 转发到后端。此处为本地开发步骤，正式发布需要HTTPS、静态文件托管、反向代理和服务托管。

## 测试

```powershell
cd backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

记忆测试入口为 `scripts/run_memory_suite.py`；先使用 `--help` 查看参数。部分测试会访问真实PG临时schema和外部模型，需配置数据库权限，并可能产生接口费用。

部分根目录测试脚本是历史调试工具，可能与当前认证协议不同；回归入口以 `tests/` 为准。测试通过不代表已经完成生产环境安全、容量或完整知识质量验收。

## 当前运行边界

- 为排障保留业务执行超时关闭的配置；生产环境需要重新确定超时、配额及容量边界。
- 流式问答当前未接入同步流程中的完整引用及覆盖度审核。
- 文档级权限、知识版本发布和完整数据库迁移仍需补齐。
- Token预算采用参考编码加安全余量，不等于DeepSeek精确计费Token。
