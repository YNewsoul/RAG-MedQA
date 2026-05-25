# 医疗 QA 知识库导入说明

本文档对应脚本：[import_medical_qa_kb.py](/c:/Users/YYXYD/Desktop/项目/RAG-MedQA/data/script/import_medical_qa_kb.py)。

目标是把 `data/medical/qa` 下的 6 个大型 JSON 问答文件，按可检索、可维护、可恢复的方式，端到端导入到本项目的 QA 知识库中，包括：

- `knowledgebase` 主表
- `document` 主表
- `file` / `file2document` 映射
- ES 主检索索引 `ragmedqa`
- 文档 metadata 索引 `ragmedqa_doc_meta`

## 1. 为什么要这样导入

这套项目真正给 `dialog` 用的不是 MySQL 原文，而是“知识库元数据 + 文档元数据 + ES chunk 索引”三部分组合。

如果只插 `knowledgebase` / `document`，但没有把标准化 chunk 写进 ES，那么：

- 对话页虽然可能看得到知识库或文档
- 但实际检索不到任何 QA 内容

所以脚本的设计原则是：

1. 先分析和清洗原始 QA 数据。
2. 再按规则切成很多个 logical documents。
3. 每个 logical document 生成一个 `document` 行。
4. 这个 document 下的每条 QA 再变成一个检索 chunk 写进 ES。

## 2. 数据来源

默认输入目录：

`data/medical/qa`

当前脚本按文件名前缀把数据分成 6 个一级大类：

- `儿科`
- `内科`
- `外科`
- `妇产科`
- `男科`
- `肿瘤科`

脚本假设每条记录是统一结构：

```json
{
  "title": "...",
  "ask": "...",
  "answer": "...",
  "department": "..."
}
```

## 3. 导入规则

### 3.1 基础清洗

每条 QA 会先做这些处理：

- `title` / `ask` / `answer` / `department` 去首尾空白
- 连续空白压成一个空格
- `ask` 或 `answer` 为空的记录直接剔除

### 3.2 去重规则

脚本按 `normalize(ask) + normalize(answer)` 做全局去重：

- 同一个文件内部的重复会被去掉
- 跨文件重复也会被去掉
- `title` 不参与去重

这样做的原因是：

- 同一问答常常会有不同标题改写
- 但对检索价值最高的是“问题+答案”主体

### 3.3 department 清洗

脚本不会直接无脑相信 `department`。

只有满足下面条件的 `department` 才会当作“干净科室名”：

- 非空
- 长度不超过 12
- 只包含中文或英文字母

否则这条记录的 `department` 会被视为脏值，不拿来独立分桶。

这样处理是因为这批数据里已经混入了类似：

- `答案：`
- `病史:...`
- 一大段回答正文

这类字符串如果直接当科室，会把分片策略带偏。

### 3.4 分片策略

脚本采用三级分片：

1. 一级：按原始文件大类分组
2. 二级：优先按 `department` 分桶
3. 三级：每个桶按约 `8000` 条 QA 切成一个 logical document

独立成桶阈值：

- 默认：`>= 2000`
- `男科`：`>= 1500`
- `肿瘤科`：`>= 1000`

不满足阈值的低频科室，以及脏 `department`，都会并入该一级大类下的 `misc` 桶。

例如：

- `qa_内科_神经科_p001`
- `qa_内科_神经科_p002`
- `qa_肿瘤科_misc_p003`

## 4. 为什么不是“一整个文件一个 document”

因为 79 万条 QA 如果只做成 6 个大 document，会带来几个问题：

- 文档粒度太粗，metadata 基本失去意义
- 后续重建成本高，改一个病种就要重建一整大块
- 文档列表和引用粒度太糙
- 排错时很难定位是哪一批数据出问题

所以这里采用：

- `1 条 QA = 1 个 chunk`
- `约 8000 条 QA = 1 个 logical document`

这样更适合这个项目的检索结构。

## 5. chunk 组织方式

每条 QA 最终会生成 1 个 ES chunk。

chunk 的核心字段包括：

- `doc_id`
- `kb_id`
- `docnm_kwd`
- `title_tks`
- `question_kwd`
- `question_tks`
- `content_ltks`
- `content_sm_ltks`
- `content_with_weight`
- `doc_type_kwd = "qa"`
- 向量字段 `q_<dim>_vec`

其中语义是：

- `question_tks` / `content_ltks`：主要偏问题面，利于问答式全文召回
- `content_with_weight`：存“标题 + 问题 + 答案”，利于向量召回和最终引用
- 向量：使用当前系统默认 embedding 模型生成

## 6. 会写入哪些地方

### 6.1 MySQL

脚本会创建或复用：

- `knowledgebase`
- `document`
- `file`
- `file2document`

这样做的好处是：

- 后台能看到知识库
- 文档列表页能看到 logical documents
- 不是只在 ES 里“裸写 chunk”

### 6.2 ES 主索引

标准化 chunk 会写入：

- `ragmedqa`

这才是 `dialog` 真正检索 QA 时使用的主索引。

### 6.3 文档 metadata 索引

每个 logical document 还会写入 metadata 文档到：

- `ragmedqa_doc_meta`

当前写入的 metadata 包括：

- `data_type`
- `major_category`
- `department`
- `bucket_type`
- `source_file`
- `shard_name`
- `planned_count`

## 7. 脚本运行阶段

脚本现在对外只保留 2 个主要阶段，外加 1 个组合阶段：

- `--phase materialize`
- `--phase import`
- `--phase all`

### 7.1 materialize

`materialize` 会在内部完成两件事：

1. 扫描原始 QA 数据，做清洗、去重和分片规划
2. 在工作目录下生成标准化分片文件

输出：

- `shard_plan.json`
- `normalized_shards/*.jsonl`
- `reports/qa_import_report.md`

其中每个 JSONL 对应一个 logical document。

### 7.2 import

`import` 不会重新扫描原始 QA 目录。  
它只消费前面已经 materialize 的 JSONL 分片和 `shard_plan.json`，然后真正写入：

- MySQL
- ES
- metadata 索引

### 7.3 all

按顺序完整执行：

`materialize -> import`

## 8. 断点续跑策略

脚本不是简单“从头插到底”，而是按 shard 粒度可恢复：

- shard JSONL 会先落到本地 workdir
- `doc_id` 按 `kb_id + shard_name` 稳定生成
- `content_hash` 会写进 `document.content_hash`
- rerun 时如果发现：
  - 文档已存在
  - hash 相同
  - `chunk_num > 0`
  - `run = DONE`
  - `progress = 1`
  - 且存在 `file2document` 映射
  
  则该 shard 会直接跳过，不重复导入

如果文档已存在但状态不完整，脚本会先清理旧文档及其 ES chunk，再重建。

## 9. 推荐运行方式

先生成分片计划和 JSONL：

```powershell
.venv\Scripts\python.exe data\script\import_medical_qa_kb.py --phase materialize
```

最后执行真正导入：

```powershell
.venv\Scripts\python.exe data\script\import_medical_qa_kb.py --phase import --kb-name medical_qa_kb
```

如果你希望一次跑完：

```powershell
.venv\Scripts\python.exe data\script\import_medical_qa_kb.py --phase all --kb-name medical_qa_kb
```

## 10. 常用参数

- `--data-dir`
  - QA 数据目录
- `--workdir`
  - `shard_plan.json`、JSONL 分片、报告输出目录
- `--kb-name`
  - 目标知识库名字
- `--limit-per-file`
  - 每个文件只取前 N 条，适合先做小规模演练
- `--embed-batch-size`
  - embedding 批量大小
- `--insert-batch-size`
  - ES 批量写入大小
- `--default-threshold`
  - 大多数大类的独立成桶阈值

## 11. 产物目录

默认 workdir：

`data/script/output/medical_qa_import`

会生成这些内容：

- `shard_plan.json`
- `normalized_shards/*.jsonl`
- `import_results.json`
- `reports/qa_import_report.md`

## 12. 注意事项

### 12.1 这不是原生“上传文件后解析”的路径

这条脚本是“旁路导入”：

- 不走前端上传
- 不走原始 JSON 文件入对象存储
- 而是直接生成 logical documents 和 ES chunks

这是有意为之，因为 79 万条 QA 更适合脚本化分片导入，而不是人手点上传。

### 12.2 document 列表里看到的是 logical documents，不是原始 6 个 JSON 文件

导入后你在后台看到的文档会是：

- `qa_内科_神经科_p001.jsonl`
- `qa_内科_神经科_p002.jsonl`
- `qa_肿瘤科_misc_p004.jsonl`

这正是脚本的目标，不是异常。

### 12.3 rerun 前不要手动改动这些 shard 的 `content_hash`

因为脚本会靠它判断某个 shard 是否已经完整导入。

### 12.4 如果默认 embedding 模型不可用，导入不会成功

脚本在导入阶段会直接调用系统当前默认 embedding 模型。

所以在真正导入前，建议先确认：

- MySQL 可连接
- ES 可连接
- embedding 模型可用

## 13. 导入完成后如何给对话使用

脚本只负责建 QA 知识库本身。

导入完成后，你还需要把该 KB 的 `kb_id` 绑定到某个 `dialog.kb_ids`，这样对话才会真正走到这个 QA 知识库。

如果后面你还要把 PDF 数据也导入，推荐继续保持“双 KB”结构：

- `medical_qa_kb`
- `medical_pdf_kb`

再让一个 `dialog` 同时绑定两个 `kb_id`。
