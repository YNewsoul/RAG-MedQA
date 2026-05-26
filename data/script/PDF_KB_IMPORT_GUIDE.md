# 医疗 PDF 知识库导入说明

## 1. 目标

这套脚本用于把 `data/medical/pdf` 目录下的医疗 PDF 文档，经过 MinerU 解析后导入到：

- MySQL：`knowledgebase / document / file / file2document`
- Elasticsearch：主 chunk 索引 `ragmedqa`
- Elasticsearch：文档 metadata 索引 `ragmedqa_doc_meta`

这套实现独立于项目原有 PDF 建库代码，入口文件是：

- `data/script/import_medical_pdf_kb.py`

核心模块在：

- `data/script/medical_pdf_import/`

## 2. 阶段设计

脚本保留和 QA 版一致的三阶段：

1. `materialize`
   - 扫描 PDF
   - 调用 MinerU 生成 markdown
   - 把 markdown 解析成脚本自己的块结构
   - 按章节/标题层级切分成 logical documents
   - 把每个 logical document 物化成 `normalized_shards/*.jsonl`

2. `import`
   - 读取 `shard_plan.json`
   - 创建/复用知识库
   - 把 JSONL 分片写入 MySQL / ES

3. `all`
   - 先 `materialize`
   - 再 `import`

## 3. 目录结构

以默认 workdir 为例：

```text
data/script/output/medical_pdf_import/
├─ pdf_manifest.json
├─ shard_plan.json
├─ materialize_cache.json
├─ mineru_markdown/
├─ parsed_blocks/
├─ normalized_shards/
├─ import_results.json
└─ reports/
   └─ pdf_import_report.md
```

各目录含义：

- `pdf_manifest.json`
  - 原始 PDF 的基础信息清单
- `materialize_cache.json`
  - 单个 PDF 的 MinerU 缓存索引
- `mineru_markdown/`
  - MinerU 输出的 markdown
- `parsed_blocks/`
  - markdown 转换后的中间块结构
- `normalized_shards/`
  - 最终供入库使用的 logical document JSONL

## 4. MinerU 调用方式

脚本支持两种方式调用 MinerU：

1. CLI 模式
   - 依赖本机能直接执行 `mineru` 或 `magic-pdf`

2. API 模式
   - 通过 HTTP POST 把 PDF 发给 MinerU 服务端

命令行控制参数：

- `--mineru-mode auto|api|cli`
- `--mineru-api-url`
- `--mineru-server-url`
- `--mineru-output-dir`
- `--mineru-parse-method auto|ocr|text|txt`

## 5. 断点续跑策略

这套脚本在 `materialize` 和 `import` 两段都做了续跑设计。

### 5.1 materialize 续跑

PDF 解析最重的是 MinerU，因此这里不是每次都重跑。

脚本会记录：

- 源 PDF 的 `source_md5`
- 对应 markdown 缓存文件
- 对应 blocks 缓存文件

只要源 PDF 没变，并且缓存文件还在：

- 下次 `materialize` 会直接复用 markdown / blocks
- 只重新生成 `normalized_shards`

如果你希望强制重新跑 MinerU，可以加：

```powershell
--force-reparse
```

### 5.2 import 续跑

这段和 QA 版一致，是“分片级幂等续跑”：

- 已成功导入的分片直接跳过
- 不完整分片先清理，再整片重导

所以断网或中断后，通常继续执行：

```powershell
.venv\Scripts\python.exe data\script\import_medical_pdf_kb.py --phase import --kb-name medical_pdf_kb_v1
```

即可续跑。

## 6. 建议命令

### 6.1 先只做 materialize

```powershell
.venv\Scripts\python.exe data\script\import_medical_pdf_kb.py --phase materialize --kb-name medical_pdf_kb_v1
```

### 6.2 再正式入库

```powershell
.venv\Scripts\python.exe data\script\import_medical_pdf_kb.py --phase import --kb-name medical_pdf_kb_v1
```

### 6.3 一次跑完

```powershell
.venv\Scripts\python.exe data\script\import_medical_pdf_kb.py --phase all --kb-name medical_pdf_kb_v1
```

### 6.4 指定 MinerU API

```powershell
.venv\Scripts\python.exe data\script\import_medical_pdf_kb.py `
  --phase materialize `
  --kb-name medical_pdf_kb_v1 `
  --mineru-mode api `
  --mineru-api-url http://127.0.0.1:9000 `
  --mineru-parse-method auto
```

## 7. 当前实现说明

当前版本的设计重点是：

- 独立脚本实现
- MinerU 预处理缓存
- 章节级 logical document 分片
- 与 QA 版一致的正式入库与续跑语义

当前版本的已知边界：

- 如果 MinerU 输出里没有可靠页码，这一版 chunk 主要保留章节来源，而不是精确页码来源
- 表格以“更适合搜索和 embedding 的纯文本形态”进入 chunk，同时保留 markdown 缓存在本地
