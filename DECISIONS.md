# 决策记录

实现过程中自己做的选择和理由。按阶段排列，后面的阶段只写新增的部分。

## 环境与配置

- **环境变量**（放在 `.env`，不进提交）：
  - 文本向量：`EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY`、`EMBEDDING_MODEL`、`EMBEDDING_DIM`、`EMBEDDING_MAX_TOKENS`（可选，默认 8192）。
  - 答案生成用的 VLM：`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`。
  - 可选：`JUDGE_MODEL`（评分器）、`RERANK_MODEL`（LLM 重排），不填就用 `LLM_MODEL`。
  - 所有远程模型都走 OpenAI 兼容接口，换服务商只改 `.env`。
- **`.env.example` 没有提交**。它现在的 `EMBEDDING_API_KEY` 一行里是一个真实样式的密钥（`sk-proj-` 开头）。AGENTS.md 规定密钥不进提交记录，所以我没有把这个文件加入任何提交，也没有改动它。需要你把密钥挪到 `.env`、把模板里的值清空后再提交。
- **两个 Python 环境**：项目 `.venv`（Python 3.12，CPU）负责除视觉编码外的全部代码和测试；`D:\rag_cache\venv_gpu` 只负责用 ColSmol 把页面和查询编码成多向量并写到磁盘。理由：GPU 环境里没有 pyarrow、pytrec_eval 等依赖，我不想改动它；编码是唯一需要 GPU 的步骤，结果落盘后其余流程都不需要 GPU。

## 阶段 1：基线

- **RAG-Anything 版本**：PyPI 最新是 1.4.4，`reference/` 里是未发布的 1.4.5。用 1.4.4，搭配 LightRAG 1.4.16。
- **不建知识图谱的做法**：LightRAG 入库时一定会做实体抽取，没有开关。给它一个返回空抽取结果的假 LLM（`NoKGLLM`），入库流程保持原样，LLM 费用为零；它的调用次数记为"省掉的 LLM 调用数"。
- **不要求安装 MinerU**：我们只插入已经解析好的 Markdown，所以跳过了 RAG-Anything 初始化时的解析器安装检查。
- **一页一个文档**：每页作为独立文档入库，`file_path` 设为 `corpus_id`，检索到的文本块能唯一对回页面。页的得分取它排名最靠前的文本块。
- **每个索引目录一个 LightRAG workspace**：LightRAG 的进程内共享状态按 workspace 名区分，不按目录区分。同一进程里先后建两个索引时，第二个会把页面误判为"已处理"而检索不到任何结果。用目录路径的哈希作 workspace 名来隔离。
- **全进程共用一个事件循环**：LightRAG 的锁和工作协程绑定在创建它们的事件循环上，所以同步接口共用一个不关闭的循环。
- **数据加载不用 `datasets` 库**：直接用 pyarrow 读 Hugging Face 缓存里的 parquet。理由：`datasets` 在离线模式下找不到按 `data_dir` 加载的缓存；直接读文件不联网、不重复下载，也不会把 1110 张页面图一次性读进内存。
- **查询抽样**：只保留指定语言且至少有一个相关页的查询，排序后用固定种子抽样，结果与数据行顺序无关。
- **语料用整个子集**（1110 页），不是只取相关页。只索引相关页会让指标虚高，也没法和后面的方案比较。
- **指标**：用 `pytrec-eval-terrier`（与 vidore-benchmark 用的 `pytrec_eval` 同一套实现，有 Windows 预编译包）。没检索到任何页的查询按 0 分计入平均。
- **答案生成没有直接调用 `aquery_vlm_enhanced`**：它的提示词来自 LightRAG 自己的检索上下文和其中的图片路径，而我们的上下文是融合后的页面列表。所以直接拼提示词：每页以 `[PAGE <id>]` 开头，后面跟页面图片或文本；要求模型输出 JSON，包含答案和引用（页 ID 加原文引文）。
- **评分器**：三档（correct / partially_correct / incorrect，对应 1 / 0.5 / 0）。解析不了的回复单独计数，不计入平均，不猜。另有导出人工抽查表和计算一致率、Cohen's kappa 的工具。
- **一个统一的运行脚本**（`mmrag.experiments.run`）取代了原来的 `baseline.py`，各管线共用同一套输出格式。
