# 20用户、单Worker、重排批次2方案

本轮已配置：

| 参数 | 当前值 |
| --- | --- |
| API Worker | 启动脚本固定1 |
| PyTorch算子线程 | 2 |
| 本地推理并发 | 1 |
| BGE重排候选 | Top20 |
| 重排批次 | 2，每个请求最多10批 |
| 查询向量缓存 | LRU最多512条；相同模型和完全相同编码文本并发只计算一次 |
| SSE总超时 | 关闭，SSE_TIMEOUT_SECONDS=0 |
| DeepSeek请求超时 | 关闭，LLM_REQUEST_TIMEOUT_SECONDS=0 |
| Jev请求与重试总时限 | 关闭，INTENT_REQUEST_TIMEOUT_SECONDS=0 |
| 工具SQL执行超时 | 关闭，TOOL_QUERY_TIMEOUT_MS=0 |
| RAG向量SQL执行超时 | 关闭，RAG_QUERY_TIMEOUT_MS=0 |
| Locust SSE总时限与读取超时 | 本轮启动脚本关闭 |

数据库连接建立超时、登录有效期、空闲超时、JWT/Tab Token/CSRF及租约监控继续保留。
服务端会话失效、客户端断连和人工Ctrl+C仍可以取消请求，并非关闭所有取消机制。
缓存只保存编码结果，不缓存答案、数据库结果或用户对话；缓存不跨服务重启保存。
缓存0容量可禁用，512条满后按最近使用顺序淘汰，异常和取消不会污染缓存。

## 启动

先在原后端终端Ctrl+C停止旧服务，避免8000端口重复实例。
终端A，在项目根目录执行（前台运行）：

```powershell
.\scripts\start_single_worker.ps1
```

确认后端启动成功。终端B，同一项目根目录执行：

```powershell
.\scripts\run_loadtest_20.ps1
```

在终端提示处输入loadtest001至loadtest020使用的密码。密码不写入文件、命令行参数或CSV。
每个用户发送一次真实知识库问题，同时独立轮询身份；本轮并不包含20次工具业务查询。
没有设置固定运行时长；20条SSE全部成功或失败并记录后自动结束。人工停止用Ctrl+C。
认证或会话创建失败会结束本轮，不能将不完整样本当作20用户通过。
结果使用带时间戳的loadtest_20_single_b2_*文件名，保留原压测CSV。
长期未结束时应看每条请求最后阶段和Worker日志，不能把等待当作成功。

## 验证与解释

34项工具/报告/图/SSE/超时/缓存离线回归通过。
另外使用合成HTTP服务验证20个不同账号的Locust客户端、无总时限和自动结束，20条全部记录成功。
真实本地缓存模型验证：重复编码只执行一次encode，返回独立副本；不调用外部LLM。
真实20用户RAG由用户选择在终端输入密码启动，以上离线成绩不是本轮真实吞吐数据。
之前3用户测试与本轮用户数、Worker、批次和缓存同时变化，不能据此单独归因于批次2。
若要严格比较批次2和4，保持其余配置一致，分别冷启动并重复同一组20用户问题。

## 后续恢复超时

本轮按用户要求关闭业务超时；恢复时可将SSE_TIMEOUT_SECONDS=300、LLM_REQUEST_TIMEOUT_SECONDS=60、
INTENT_REQUEST_TIMEOUT_SECONDS=15、TOOL_QUERY_TIMEOUT_MS=30000、RAG_QUERY_TIMEOUT_MS=15000。
启动脚本也显式指定这些参数，恢复时需同步调整启动方式，不能只修改.env。
