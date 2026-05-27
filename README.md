# 医疗RAG问诊系统（RAG-MedQA）

面向医疗问答场景的 RAG 系统，支持：

- 中文医疗问答数据集建库
- 医疗书籍 / 临床指南 PDF 建库
- 混合检索（全文 + 向量）与重排
- SSE 流式回答与引用溯源
- 离线检索评估与参数搜索

当前仓库已经包含两套独立的数据导入脚本：

- `data/script/import_medical_qa_kb.py`
- `data/script/import_medical_pdf_kb.py`

它们直接落到项目现有的 MySQL / Elasticsearch / 文件树元数据体系中，适合你当前这类医疗 QA 与 PDF 数据的端到端建库。

## 目录

- [系统架构](#系统架构)
- [系统概览](#系统概览)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [知识库构建](#知识库构建)
- [配置说明](#配置说明)
- [开发指南](#开发指南)
- [主要接口](#主要接口)
- [评估](#评估)
- [免责声明](#免责声明)

## 系统架构

```text
┌─────────────────────────────────────────────────────────────┐
│                      前端界面（React SPA）                    │
│         登录 / 会话切换 / SSE 渲染 / 引用展示 / 知识库交互       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / SSE
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Quart 后端（api/）                       │
│      用户认证 / 对话管理 / 检索编排 / Prompt 组装 / LLM 调用     │
└───────────────┬──────────────────┬────────────────┬──────────┘
                │                  │                │
                ▼                  ▼                ▼
      ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
      │   MySQL        │  │ Elasticsearch  │  │ Redis / MinIO  │
      │ 用户/对话/KB元数据 │  │ chunk索引/文档元数据 │  │ 锁/缓存/对象存储 │
      └────────────────┘  └────────────────┘  └────────────────┘
                │
                ▼
      ┌────────────────────────────────────────────────────────┐
      │                    RAG 核心（rag/）                    │
      │ 全文检索 / 向量检索 / 重排 / 引用插入 / Prompt 注入      │
      └────────────────────────────────────────────────────────┘
```

当前仓库里，知识库构建主要有两条独立链路：

- `data/script/import_medical_qa_kb.py`：面向医疗 QA 数据
- `data/script/import_medical_pdf_kb.py`：面向医疗 PDF 数据

它们最终都会落到同一套运行时体系中：

- MySQL：`knowledgebase / document / file / file2document / dialog`
- Elasticsearch：`ragmedqa` 与 `ragmedqa_doc_meta`
- 前端问答：通过 `dialog.kb_ids + prompt_config` 进入统一 RAG 检索链路

## 系统概览

系统由 4 个主要部分组成：

1. 前端 SPA  
   位于 `web/`，负责对话、会话切换、SSE 渲染、引用展示。

2. Quart 后端  
   位于 `api/`，负责用户、对话、检索编排、LLM 调用与流式输出。

3. RAG 检索与 Prompt 组装  
   位于 `rag/`，负责全文检索、向量检索、重排、引用与提示词拼装。

4. 数据与基础设施  
   - MySQL：用户、对话、知识库、文档、文件树元数据
   - Elasticsearch：chunk 主索引与文档 metadata 索引
   - Redis：锁、任务状态
   - MinIO：对象存储

## 项目结构

```text
RAG-MedQA/
├─ api/                         # Quart 后端
│  ├─ apps/                     # 路由与应用入口
│  ├─ db/                       # Peewee 模型与业务服务
│  └─ ragflow_server.py         # 服务启动入口
├─ rag/                         # RAG 核心
│  ├─ nlp/                      # query/search/term_weight 等
│  ├─ llm/                      # chat/embedding/rerank 抽象层
│  └─ prompts/                  # prompt 模板与组装逻辑
├─ parser/                      # 文档解析器
├─ web/                         # React + TypeScript 前端
├─ common/                      # 共享配置与工具
├─ conf/                        # 主配置与索引 mapping
├─ docker/                      # Docker Compose 与 Nginx 配置
├─ evaluation/                  # 离线检索评估工具
├─ data/script/                 # 自定义 QA / PDF 建库脚本
│  ├─ import_medical_qa_kb.py
│  ├─ import_medical_pdf_kb.py
│  ├─ medical_qa_import/
│  └─ medical_pdf_import/
└─ test/                        # benchmark / 单测 / e2e
```

## 快速开始

### 环境要求

- Python `>=3.12,<3.15`
- Node.js `>=18.20.4`
- Docker + Docker Compose
- 建议内存 `16GB+`
- 建议磁盘 `30GB+`

### 生产部署

```bash
# 1. 准备环境变量
cp docker/.env.example docker/.env

# 2. 根据需要修改 docker/.env

# 3. 启动完整服务
cd docker
docker compose up -d

# 4. 访问
# Web UI: http://localhost
# API:    http://localhost/api/v1
```

### 开发环境

#### 1. 启动基础设施

```bash
docker compose -f docker/docker-compose-base.yml up -d
```

这会启动：

- MySQL
- Redis
- Elasticsearch
- MinIO

#### 2. 启动后端

```bash
uv sync --python 3.12
python api/ragflow_server.py
```

后端默认端口：

- API：`http://localhost:9380`
- Admin：`http://localhost:9381`

#### 3. 启动前端

```bash
cd web
npm install
npm run dev
```

前端开发服务器默认会监听 Vite 本地地址。

### 推荐的本地覆盖配置

不要直接修改主配置 `conf/service_conf.yaml`。  
建议新增本地覆盖文件：

- `conf/local.service_conf.yaml`

常见用途：

- 切换聊天模型
- 切换 embedding 服务地址
- 切换 reranker
- 使用本地或远端隧道地址

例如：

```yaml
user_default_llm:
  default_models:
    embedding_model:
      name: 'BAAI/bge-m3'
      factory: 'OpenAI-API-Compatible'
      api_key: '-'
      base_url: 'http://localhost:6381'
```

## 知识库构建

### 1. QA 知识库

入口：

- [data/script/import_medical_qa_kb.py](data/script/import_medical_qa_kb.py)
- [data/script/QA_KB_IMPORT_GUIDE.md](data/script/QA_KB_IMPORT_GUIDE.md)

推荐流程：

```bash
python data/script/import_medical_qa_kb.py --phase materialize
python data/script/import_medical_qa_kb.py --phase import --kb-name medical_qa_kb_v1
```

或一步跑完：

```bash
python data/script/import_medical_qa_kb.py --phase all --kb-name medical_qa_kb_v1
```

这套脚本会完成：

- 原始 QA 清洗、去重、分桶、分片
- `normalized_shards/*.jsonl` 物化
- `knowledgebase / document / file / file2document` 写入
- chunk 写入 ES 主索引
- 分片级幂等续跑

### 2. PDF 知识库

入口：

- [data/script/import_medical_pdf_kb.py](data/script/import_medical_pdf_kb.py)
- [data/script/PDF_KB_IMPORT_GUIDE.md](data/script/PDF_KB_IMPORT_GUIDE.md)

这套脚本与仓库原有 PDF 处理流程解耦，适合当前医疗书籍 / 指南类 PDF。

支持：

- MinerU `CLI / API / auto`
- Markdown 缓存
- 标题 / 段落 / HTML 表格 / Markdown 表格解析
- logical document 分片
- MySQL + ES 入库
- 分片级幂等续跑

推荐流程：

```bash
python data/script/import_medical_pdf_kb.py \
  --phase materialize \
  --kb-name medical_pdf_kb_v1 \
  --mineru-mode api \
  --mineru-api-url http://127.0.0.1:8003 \
  --mineru-parse-method auto

python data/script/import_medical_pdf_kb.py \
  --phase import \
  --kb-name medical_pdf_kb_v1 \
  --workdir data/script/output/medical_pdf_import
```

或一步跑完：

```bash
python data/script/import_medical_pdf_kb.py \
  --phase all \
  --kb-name medical_pdf_kb_v1 \
  --mineru-mode api \
  --mineru-api-url http://127.0.0.1:8003 \
  --mineru-parse-method auto
```

### 3. 建库后的对话接入

对话真正触发 RAG，需要同时满足：

1. `dialog.kb_ids` 绑定知识库
2. `prompt_config.parameters` 包含 `knowledge`
3. `prompt_config.system` 中包含 `{knowledge}`

否则即使绑了 KB，也可能不会进入知识库检索分支。

### 4. 续跑说明

两套导入脚本都支持分片级幂等续跑：

- 已成功的 shard 会跳过
- 半成品 shard 会清理后重导
- `kb-name` 与 `workdir` 不变即可续跑

## 配置说明

### 主配置

主配置文件：

- [conf/service_conf.yaml](conf/service_conf.yaml)

主要内容包括：

- MySQL
- Elasticsearch
- Redis
- MinIO
- 默认聊天模型
- 默认 embedding 模型
- 默认 reranker

### Elasticsearch

当前代码默认文档引擎走 Elasticsearch。  
主 chunk 索引名为：

- `ragmedqa`

文档 metadata 索引名为：

- `ragmedqa_doc_meta`

字段 mapping 定义在：

- [conf/mapping.json](conf/mapping.json)

### LLM / Embedding / Reranker

模型抽象层位于：

- `rag/llm/chat_model.py`
- `rag/llm/embedding_model.py`
- `rag/llm/rerank_model.py`

如果你使用 OpenAI 兼容 embedding 服务，推荐通过 `local.service_conf.yaml` 覆盖 `base_url`。

## 开发指南

### 后端

```bash
uv sync --python 3.12
python api/ragflow_server.py --debug
```

初始化超级用户：

```bash
python api/ragflow_server.py --init-superuser
```

### 前端

```bash
cd web
npm install
npm run dev
npm run build
```

### 核心文件

- 对话主链路：`api/db/services/dialog_service.py`
- 检索与重排：`rag/nlp/search.py`
- 查询构造：`rag/nlp/query.py`
- Prompt 组装：`rag/prompts/generator.py`
- QA 建库脚本：`data/script/medical_qa_import/`
- PDF 建库脚本：`data/script/medical_pdf_import/`

## 主要接口

### 用户

- `POST /v1/user/login`
- `POST /v1/user/register`
- `GET /v1/user/info`

### 对话 / 会话

- `GET /api/v1/chats`
- `POST /api/v1/chats`
- `GET /api/v1/chats/{chat_id}`
- `PUT /api/v1/chats/{chat_id}`
- `DELETE /api/v1/chats/{chat_id}`
- `GET /api/v1/chats/{chat_id}/sessions`
- `POST /api/v1/chats/{chat_id}/sessions`
- `GET /api/v1/chats/{chat_id}/sessions/{id}`
- `PUT /api/v1/chats/{chat_id}/sessions/{id}`
- `POST /api/v1/chats/ask`

### SSE 问答

`/api/v1/chats/ask` 支持 SSE 流式返回。

典型响应分为：

- 增量帧：`final = false`
- 最终帧：`final = true`
- 最终帧中会带完整答案与引用信息

## 评估

### 1. 离线检索评估

目录：

- [evaluation/](evaluation)

主要脚本：

- [evaluation/build_dataset.py](evaluation/build_dataset.py)
- [evaluation/run_eval.py](evaluation/run_eval.py)
- [evaluation/grid_search.py](evaluation/grid_search.py)

运行方式：

```bash
# 1. 构建评估数据集
python evaluation/build_dataset.py

# 2. 运行检索评估
python evaluation/run_eval.py

# 3. 参数网格搜索
python evaluation/grid_search.py
```

结果输出到：

- `evaluation/results/`

重点指标通常包括：

- `MRR`
- `Recall@K`
- `Top1 / TopN`

### 2. HTTP Benchmark

目录：

- `test/benchmark/`

适合压测：

- chat SSE 延迟
- retrieval 延迟
- 吞吐与接口稳定性

## 免责声明

本项目面向医疗信息检索与问答研究、原型开发与工程验证场景。  
系统输出仅供参考，不构成正式诊疗建议。涉及真实医疗决策时，请以执业医师的专业判断为准。
