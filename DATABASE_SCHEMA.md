# RAG-MedQA 数据库结构说明

## 1. 文档范围

这份文档基于两部分内容整理：

- 代码中的模型定义：`api/db/db_models.py`
- 你当前这套本地数据库的实际状态

文档目标是帮助你快速理解：

- 项目一共有哪些表
- 每张表是做什么的
- 主要字段分别表示什么
- 你当前环境里哪些表已经有数据，哪些还没有

## 2. 当前数据库快照

当前项目连接的是 MySQL，配置文件在 `conf/service_conf.yaml`。  
从模型定义和实际数据库来看，项目当前一共创建了 **24 张表**。

### 2.1 你当前数据库中的表和行数

| 表名 | 当前行数 | 说明 |
|---|---:|---|
| `api_4_conversation` | 0 | 扩展会话表 |
| `api_token` | 0 | 外部 API token 绑定表 |
| `app_config` | 0 | 通用键值配置表 |
| `canvas_template` | 0 | 画布模板表 |
| `connector` | 0 | 外部数据连接器配置表 |
| `connector2kb` | 0 | 连接器与知识库映射表 |
| `conversation` | 0 | 普通聊天会话表 |
| `dialog` | 0 | Chat Assistant / 对话应用主表 |
| `document` | 0 | 知识库文档表 |
| `file` | 1 | 文件树表，目前只有根目录 |
| `file2document` | 0 | 文件与文档映射表 |
| `invitation_code` | 0 | 邀请码表 |
| `knowledgebase` | 0 | 知识库主表 |
| `llm` | 887 | 模型目录表，已初始化 |
| `llm_factories` | 59 | 模型供应商目录表，已初始化 |
| `mcp_server` | 0 | MCP Server 配置表 |
| `memory` | 0 | 长期记忆配置表 |
| `pipeline_operation_log` | 0 | 流水线操作日志表 |
| `search` | 0 | 检索应用配置表 |
| `sync_logs` | 0 | 连接器同步日志表 |
| `system_settings` | 14 | 系统设置表，已初始化 |
| `task` | 0 | 文档处理任务表 |
| `user` | 1 | 用户表，已有 1 个普通用户 |
| `user_canvas` | 0 | 用户画布表 |
| `user_canvas_version` | 0 | 画布版本表 |

### 2.2 对当前状态的解释

你现在这套库已经完成了“系统级初始化”，但还没有形成“业务级数据”。

已经初始化好的主要是：

- `llm_factories`
- `llm`
- `system_settings`

已经开始有用户侧数据的表：

- `user`
- `file`

仍然为空、说明业务内容还没创建的关键表：

- `dialog`
- `conversation`
- `knowledgebase`
- `document`

这也是你之前前端写死 `DIALOG_ID` 后报错的根本原因：  
**`dialog` 表当前是 0 行，所以任何写死的 `DIALOG_ID` 都找不到对应记录。**

## 3. 通用约定

## 3.1 公共审计字段

绝大多数表都继承了这 4 个公共字段：

| 字段名 | 类型 | 含义 |
|---|---|---|
| `create_time` | `BigIntegerField` | 创建时间戳，毫秒 |
| `create_date` | `DateTimeField` | 创建时间，日期时间格式 |
| `update_time` | `BigIntegerField` | 最后更新时间戳，毫秒 |
| `update_date` | `DateTimeField` | 最后更新时间，日期时间格式 |

后面的逐表说明里，不再每次重复解释这 4 个字段。

## 3.2 主键风格

- 大多数表都使用 `CharField(32)` 作为主键
- 这些 ID 是逻辑字符串 ID，不是自增整数
- `llm` 使用真正的多列主键：`fid + llm_name`
- `api_token` 虽然代码里用了 `CompositeKey`，但实际只有 `token` 一个字段

## 3.3 常见状态字段

很多表会有 `status` 字段，一般约定是：

- `status = "1"`：有效、启用、可用
- `status = "0"`：逻辑删除、失效、废弃

## 3.4 JSON 字段的存储方式

项目里自定义了 `JSONField`。在 MySQL 里它本质上会以文本形式保存，但在 Python 层读出来会自动转成对象或数组。

这类字段包括：

- `prompt_config`
- `llm_setting`
- `parser_config`
- `message`
- `reference`
- `search_config`
- `dsl`

所以你在数据库里直接看这些字段，通常会看到 JSON 字符串。

## 3.5 关系以逻辑 ID 为主，而不是强外键

这个项目虽然有很多“关联关系”，但大多数都不是数据库层面的强外键约束，而是业务代码按 ID 进行关联。

例如：

- `document.kb_id` 对应 `knowledgebase.id`
- `conversation.dialog_id` 对应 `dialog.id`
- `file2document.file_id` 对应 `file.id`
- `connector2kb.connector_id` 对应 `connector.id`

这意味着：

- 结构更灵活
- 但排查问题时要更依赖代码逻辑，不能只看数据库约束

## 4. 当前非空表说明

## 4.1 `user`

目前有 1 条记录，是一个普通用户，不是超级管理员。

这说明：

- 你已经注册/创建过用户
- 登录系统的用户侧链路基本是通的

## 4.2 `file`

目前有 1 条记录，是根目录节点。

它的表现说明：

- 文件树结构已经初始化
- 但还没有真正上传文件进入系统

## 4.3 `system_settings`

目前有 14 条配置，主要包括：

- 白名单开关
- 邮件发送配置
- 代码沙箱配置

这些配置来自 `conf/system_settings.json`，是启动初始化时自动插入的。

## 4.4 `llm_factories` 和 `llm`

这两张表已经有数据，说明模型供应商目录和模型目录已经被初始化好了。

它们属于“系统参考数据”，不是你的业务内容。

换句话说：

- 模型目录存在
- 但聊天应用 `dialog` 还没创建

## 5. 分域逐表说明

## 5.1 用户与认证域

### `user`

- 用途：用户主表，保存账号、密码哈希、展示偏好、管理员标记等
- 主键：`id`
- 当前行数：`1`
- 逻辑关系：
  - 被 `file.created_by` 引用
  - 被 `knowledgebase.created_by` 引用
  - 被 `conversation.user_id` 引用
  - 被 `user_canvas.user_id` 引用

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 用户 ID |
| `access_token` | `CharField` | 当前用户 access token |
| `nickname` | `CharField` | 昵称/显示名 |
| `password` | `CharField` | 密码哈希 |
| `email` | `CharField` | 用户邮箱，唯一 |
| `avatar` | `TextField` | 用户头像，通常是 base64 |
| `language` | `CharField` | 语言偏好，如 `Chinese` / `English` |
| `color_schema` | `CharField` | 主题偏好，如 `Bright` / `Dark` |
| `timezone` | `CharField` | 时区配置 |
| `last_login_time` | `DateTimeField` | 上次登录时间 |
| `is_authenticated` | `CharField` | 认证标记 |
| `is_active` | `CharField` | 是否启用 |
| `is_anonymous` | `CharField` | 是否匿名账号 |
| `login_channel` | `CharField` | 登录方式，如 `password`、OAuth 等 |
| `status` | `CharField` | 逻辑有效标记 |
| `is_superuser` | `BooleanField` | 是否为超级管理员 |

### `invitation_code`

- 用途：邀请码记录表
- 主键：`id`
- 当前行数：`0`
- 逻辑关系：
  - `user_id` 指向 `user`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 邀请记录 ID |
| `code` | `CharField` | 邀请码字符串 |
| `visit_time` | `DateTimeField` | 使用/访问时间 |
| `user_id` | `CharField` | 关联用户 ID |
| `status` | `CharField` | 邀请码状态 |

### `api_token`

- 用途：外部 API token 与对话应用的绑定表
- 主键：`token`
- 当前行数：`0`
- 逻辑关系：
  - `dialog_id` 指向 `dialog.id`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `token` | `CharField` | API token 值 |
| `dialog_id` | `CharField` | 绑定的对话应用 ID |
| `source` | `CharField` | token 来源，常见是 `none`、`agent`、`dialog` |
| `beta` | `CharField` | 预留扩展字段 |

### `app_config`

- 用途：简单应用配置表
- 主键：`key`
- 当前行数：`0`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `key` | `CharField` | 配置键 |
| `value` | `JSONField` | 配置值，JSON 形式 |

### `system_settings`

- 用途：系统级设置表
- 主键：`name`
- 当前行数：`14`
- 数据来源：`conf/system_settings.json`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `name` | `CharField` | 配置名 |
| `source` | `CharField` | 配置来源，当前多为 `variable` |
| `data_type` | `CharField` | 值类型，如 `string`、`bool`、`integer`、`json` |
| `value` | `TextField` | 配置值 |

## 5.2 模型与平台扩展域

### `llm_factories`

- 用途：模型供应商目录表
- 主键：`name`
- 当前行数：`59`
- 逻辑关系：
  - 被 `llm.fid` 引用

| 字段名 | 类型 | 含义 |
|---|---|---|
| `name` | `CharField` | 供应商名，如 DeepSeek、OpenAI、Qwen 等 |
| `logo` | `TextField` | 供应商图标 |
| `tags` | `CharField` | 能力标签 |
| `rank` | `IntegerField` | 排序优先级 |
| `status` | `CharField` | 逻辑有效标记 |

### `llm`

- 用途：模型目录表，记录每个供应商下的具体模型
- 主键：`fid + llm_name`
- 当前行数：`887`
- 逻辑关系：
  - `knowledgebase.embd_id` 会引用它
  - `dialog.llm_id` 会引用它
  - `dialog.rerank_id` 会引用它
  - `memory.llm_id` / `memory.embd_id` 会引用它

| 字段名 | 类型 | 含义 |
|---|---|---|
| `llm_name` | `CharField` | 模型名 |
| `model_type` | `CharField` | 模型类型，如 Chat、Embedding、Image2Text、ASR |
| `fid` | `CharField` | 供应商 ID，对应 `llm_factories.name` |
| `max_tokens` | `IntegerField` | 模型 token 上限元数据 |
| `tags` | `CharField` | 模型能力标签 |
| `is_tools` | `BooleanField` | 是否支持工具调用 |
| `status` | `CharField` | 逻辑有效标记 |

### `mcp_server`

- 用途：MCP Server 配置表
- 主键：`id`
- 当前行数：`0`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | MCP Server ID |
| `name` | `CharField` | MCP Server 名称 |
| `url` | `CharField` | MCP Server 地址 |
| `server_type` | `CharField` | MCP Server 类型 |
| `description` | `TextField` | 描述 |
| `variables` | `JSONField` | 变量配置 |
| `headers` | `JSONField` | 附加请求头 |

## 5.3 知识库与文档处理域

### `knowledgebase`

- 用途：知识库主表
- 主键：`id`
- 当前行数：`0`
- 逻辑关系：
  - 被 `document.kb_id` 引用
  - 被 `connector2kb.kb_id` 引用
  - 被 `pipeline_operation_log.kb_id` 引用
  - 被 `sync_logs.kb_id` 引用

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 知识库 ID |
| `avatar` | `TextField` | 知识库头像/图标 |
| `name` | `CharField` | 知识库名称 |
| `language` | `CharField` | 知识库语言 |
| `description` | `TextField` | 描述 |
| `embd_id` | `CharField` | 默认向量模型 ID |
| `permission` | `CharField` | 权限范围，通常是 `me` 或 `team` |
| `created_by` | `CharField` | 创建者用户 ID |
| `doc_num` | `IntegerField` | 文档数量 |
| `token_num` | `IntegerField` | 总 token 数 |
| `chunk_num` | `IntegerField` | 总 chunk 数 |
| `similarity_threshold` | `FloatField` | 相似度阈值 |
| `vector_similarity_weight` | `FloatField` | 向量相似度权重 |
| `parser_id` | `CharField` | 默认解析器 ID |
| `pipeline_id` | `CharField` | 流水线 ID |
| `parser_config` | `JSONField` | 解析器配置 |
| `pagerank` | `IntegerField` | 排序权重 |
| `graphrag_task_id` | `CharField` | GraphRAG 任务 ID |
| `graphrag_task_finish_at` | `DateTimeField` | GraphRAG 完成时间 |
| `raptor_task_id` | `CharField` | RAPTOR 任务 ID |
| `raptor_task_finish_at` | `DateTimeField` | RAPTOR 完成时间 |
| `mindmap_task_id` | `CharField` | Mindmap 任务 ID |
| `mindmap_task_finish_at` | `DateTimeField` | Mindmap 完成时间 |
| `status` | `CharField` | 逻辑有效标记 |

### `document`

- 用途：知识库中的文档表
- 主键：`id`
- 当前行数：`0`
- 逻辑关系：
  - `kb_id` 指向 `knowledgebase`
  - 通过 `file2document` 可映射到 `file`
  - 文档处理任务会写入 `task`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 文档 ID |
| `thumbnail` | `TextField` | 缩略图 |
| `kb_id` | `CharField` | 所属知识库 ID |
| `parser_id` | `CharField` | 解析器 ID |
| `pipeline_id` | `CharField` | 流水线 ID |
| `parser_config` | `JSONField` | 解析器配置 |
| `source_type` | `CharField` | 文档来源类型，如 `local` |
| `type` | `CharField` | 文件扩展名/文档类型 |
| `created_by` | `CharField` | 创建者用户 ID |
| `name` | `CharField` | 文件名 |
| `location` | `CharField` | 文件存储位置 |
| `size` | `IntegerField` | 文件大小 |
| `token_num` | `IntegerField` | token 数 |
| `chunk_num` | `IntegerField` | chunk 数 |
| `progress` | `FloatField` | 处理进度 |
| `progress_msg` | `TextField` | 处理说明 |
| `process_begin_at` | `DateTimeField` | 处理开始时间 |
| `process_duration` | `FloatField` | 处理耗时 |
| `suffix` | `CharField` | 实际文件后缀 |
| `content_hash` | `CharField` | 内容哈希，用于变更检测 |
| `run` | `CharField` | 处理开关/取消标记 |
| `status` | `CharField` | 逻辑有效标记 |

### `file`

- 用途：文件中心 / 文件树表
- 主键：`id`
- 当前行数：`1`
- 逻辑关系：
  - `parent_id` 构成目录树
  - 可通过 `file2document` 与 `document` 关联

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 文件节点 ID |
| `parent_id` | `CharField` | 父目录 ID；根目录会指向自己 |
| `created_by` | `CharField` | 创建者用户 ID |
| `name` | `CharField` | 文件名或目录名 |
| `location` | `CharField` | 文件物理存储位置 |
| `size` | `IntegerField` | 文件大小 |
| `type` | `CharField` | 类型，如 `folder`、`pdf`、`json` |
| `source_type` | `CharField` | 来源类型 |

### `file2document`

- 用途：原始文件和解析后文档之间的映射表
- 主键：`id`
- 当前行数：`0`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 映射记录 ID |
| `file_id` | `CharField` | 文件 ID |
| `document_id` | `CharField` | 文档 ID |

### `task`

- 用途：文档处理任务表
- 主键：`id`
- 当前行数：`0`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 任务 ID |
| `doc_id` | `CharField` | 对应文档 ID |
| `from_page` | `IntegerField` | 起始页 |
| `to_page` | `IntegerField` | 结束页 |
| `task_type` | `CharField` | 任务类型 |
| `priority` | `IntegerField` | 优先级 |
| `begin_at` | `DateTimeField` | 开始时间 |
| `process_duration` | `FloatField` | 处理耗时 |
| `progress` | `FloatField` | 进度 |
| `progress_msg` | `TextField` | 进度说明 |
| `retry_count` | `IntegerField` | 重试次数 |
| `digest` | `TextField` | 任务摘要 |
| `chunk_ids` | `LongTextField` | 相关 chunk ID 列表 |

### `pipeline_operation_log`

- 用途：流水线/解析任务的操作日志表
- 主键：`id`
- 当前行数：`0`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 日志 ID |
| `document_id` | `CharField` | 文档 ID |
| `kb_id` | `CharField` | 知识库 ID |
| `pipeline_id` | `CharField` | 流水线 ID |
| `pipeline_title` | `CharField` | 流水线标题 |
| `parser_id` | `CharField` | 解析器 ID |
| `document_name` | `CharField` | 文档名 |
| `document_suffix` | `CharField` | 文档后缀 |
| `document_type` | `CharField` | 文档类型 |
| `source_from` | `CharField` | 来源 |
| `progress` | `FloatField` | 进度 |
| `progress_msg` | `TextField` | 进度说明 |
| `process_begin_at` | `DateTimeField` | 开始时间 |
| `process_duration` | `FloatField` | 耗时 |
| `dsl` | `JSONField` | 流水线 DSL 快照 |
| `task_type` | `CharField` | 任务类型 |
| `operation_status` | `CharField` | 操作状态 |
| `avatar` | `TextField` | 图标 |
| `status` | `CharField` | 逻辑有效标记 |

### `connector`

- 用途：外部数据连接器配置表
- 主键：`id`
- 当前行数：`0`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 连接器 ID |
| `name` | `CharField` | 连接器名称 |
| `source` | `CharField` | 数据源类型 |
| `input_type` | `CharField` | 输入方式，如 `poll`、`event` |
| `config` | `JSONField` | 连接器配置 |
| `refresh_freq` | `IntegerField` | 刷新频率 |
| `prune_freq` | `IntegerField` | 清理频率 |
| `timeout_secs` | `IntegerField` | 超时时间 |
| `indexing_start` | `DateTimeField` | 索引开始时间 |
| `status` | `CharField` | 调度/状态字段 |

### `connector2kb`

- 用途：连接器与知识库的绑定关系表
- 主键：`id`
- 当前行数：`0`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 映射 ID |
| `connector_id` | `CharField` | 连接器 ID |
| `kb_id` | `CharField` | 知识库 ID |
| `auto_parse` | `CharField` | 是否自动解析导入文档 |

### `sync_logs`

- 用途：连接器同步日志表
- 主键：`id`
- 当前行数：`0`
- 特殊说明：
  - `poll_range_start` 和 `poll_range_end` 使用自定义 `DateTimeTzField`
  - 在 MySQL 中它们会以 ISO8601 字符串的形式存储在 `VARCHAR` 里

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 同步日志 ID |
| `connector_id` | `CharField` | 连接器 ID |
| `status` | `CharField` | 处理状态 |
| `from_beginning` | `CharField` | 是否从头开始同步 |
| `new_docs_indexed` | `IntegerField` | 新增索引文档数 |
| `total_docs_indexed` | `IntegerField` | 累计索引文档数 |
| `docs_removed_from_index` | `IntegerField` | 从索引中移除的文档数 |
| `error_msg` | `TextField` | 错误摘要 |
| `error_count` | `IntegerField` | 错误数量 |
| `full_exception_trace` | `TextField` | 完整异常堆栈 |
| `time_started` | `DateTimeField` | 同步开始时间 |
| `poll_range_start` | `DateTimeTzField` | 轮询时间窗口开始 |
| `poll_range_end` | `DateTimeTzField` | 轮询时间窗口结束 |
| `kb_id` | `CharField` | 目标知识库 ID |

## 5.4 对话、检索与记忆域

### `dialog`

- 用途：Chat Assistant / 对话应用主表
- 主键：`id`
- 当前行数：`0`
- 逻辑关系：
  - 被 `conversation.dialog_id` 引用
  - 被 `api_4_conversation.dialog_id` 引用
  - 被 `api_token.dialog_id` 引用

最重要的现实意义：

- 前端写死的 `DIALOG_ID` 理论上就应该对应这里的一条记录
- 你现在这张表是空的，所以写死 ID 一定会失败

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 对话应用 / Chat Assistant ID |
| `name` | `CharField` | 对话应用名称 |
| `description` | `TextField` | 描述 |
| `icon` | `TextField` | 图标 |
| `language` | `CharField` | 语言 |
| `llm_id` | `CharField` | 默认聊天模型 ID |
| `tenant_llm_id` | `IntegerField` | 绑定的租户模型配置 ID |
| `tenant_rerank_id` | `IntegerField` | 绑定的租户重排模型配置 ID |
| `llm_setting` | `JSONField` | 推理参数，如 `temperature`、`top_p`、`max_tokens` |
| `prompt_type` | `CharField` | 提示词模式，如 `simple`、`advanced` |
| `prompt_config` | `JSONField` | 提示词配置，常见包含 `system`、`prologue`、`parameters`、`empty_response` |
| `meta_data_filter` | `JSONField` | 元数据过滤配置 |
| `similarity_threshold` | `FloatField` | 检索阈值 |
| `vector_similarity_weight` | `FloatField` | 向量相似度权重 |
| `top_n` | `IntegerField` | 最终引用/使用的 chunk 数 |
| `top_k` | `IntegerField` | 初筛候选数量 |
| `do_refer` | `CharField` | 回答中是否插入引用标记 |
| `rerank_id` | `CharField` | 默认重排模型 ID |
| `kb_ids` | `JSONField` | 绑定的知识库 ID 列表 |
| `status` | `CharField` | 逻辑有效标记 |

### `conversation`

- 用途：普通聊天会话表
- 主键：`id`
- 当前行数：`0`
- 逻辑关系：
  - `dialog_id` 指向 `dialog`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 会话 ID |
| `dialog_id` | `CharField` | 所属对话应用 ID |
| `name` | `CharField` | 会话标题 |
| `message` | `JSONField` | 消息历史 |
| `reference` | `JSONField` | 引用/溯源历史 |
| `user_id` | `CharField` | 所属用户 ID |

### `api_4_conversation`

- 用途：API/iframe/agent 场景下的扩展会话表
- 主键：`id`
- 当前行数：`0`
- 和 `conversation` 的区别：
  - 多了 `tokens`、`duration`、`round`、`errors`、`dsl` 等运行态信息

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 会话 ID |
| `name` | `CharField` | 会话标题 |
| `dialog_id` | `CharField` | 所属对话应用 ID |
| `user_id` | `CharField` | 所属用户 ID |
| `exp_user_id` | `CharField` | 扩展用户 ID |
| `message` | `JSONField` | 消息历史 |
| `reference` | `JSONField` | 引用历史 |
| `tokens` | `IntegerField` | token 用量 |
| `source` | `CharField` | 来源类型 |
| `dsl` | `JSONField` | 关联流程 DSL 快照 |
| `duration` | `FloatField` | 耗时 |
| `round` | `IntegerField` | 轮次 |
| `thumb_up` | `IntegerField` | 点赞数 |
| `errors` | `TextField` | 错误信息 |
| `version_title` | `CharField` | 创建会话时使用的画布版本标题 |

### `search`

- 用途：检索应用配置表
- 主键：`id`
- 当前行数：`0`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 检索应用 ID |
| `avatar` | `TextField` | 图标 |
| `name` | `CharField` | 检索应用名称 |
| `description` | `TextField` | 描述 |
| `created_by` | `CharField` | 创建者用户 ID |
| `search_config` | `JSONField` | 检索配置，包含知识库范围、文档范围、阈值、重排、摘要、网页搜索等 |
| `status` | `CharField` | 逻辑有效标记 |

### `memory`

- 用途：长期记忆配置表
- 主键：`id`
- 当前行数：`0`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 记忆配置 ID |
| `name` | `CharField` | 记忆名称 |
| `avatar` | `TextField` | 图标 |
| `memory_type` | `IntegerField` | 位标记：`1=raw`、`2=semantic`、`4=episodic`、`8=procedural` |
| `storage_type` | `CharField` | 存储类型，如 `table` 或 `graph` |
| `embd_id` | `CharField` | Embedding 模型 ID |
| `llm_id` | `CharField` | Chat 模型 ID |
| `permissions` | `CharField` | 权限范围 |
| `description` | `TextField` | 描述 |
| `memory_size` | `IntegerField` | 记忆容量上限 |
| `forgetting_policy` | `CharField` | 遗忘策略，如 `FIFO` 或 `LRU` |
| `temperature` | `FloatField` | 生成温度 |
| `system_prompt` | `TextField` | 系统提示词 |
| `user_prompt` | `TextField` | 用户提示词 |

## 5.5 画布与工作流域

### `user_canvas`

- 用途：用户自定义画布表，承载 Agent/Dataflow 画布
- 主键：`id`
- 当前行数：`0`
- 逻辑关系：
  - `user_id` 指向 `user`
  - 被 `user_canvas_version.user_canvas_id` 引用

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 画布 ID |
| `avatar` | `TextField` | 画布图标 |
| `user_id` | `CharField` | 所属用户 ID |
| `title` | `CharField` | 画布标题 |
| `permission` | `CharField` | 权限范围 |
| `release` | `BooleanField` | 是否发布 |
| `description` | `TextField` | 描述 |
| `canvas_type` | `CharField` | 画布类型 |
| `canvas_category` | `CharField` | 画布分类，如 `agent_canvas`、`dataflow_canvas` |
| `dsl` | `JSONField` | 画布 DSL |

### `canvas_template`

- 用途：系统内置画布模板表
- 主键：`id`
- 当前行数：`0`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 模板 ID |
| `avatar` | `TextField` | 模板图标 |
| `title` | `JSONField` | 模板标题，支持多语言 |
| `description` | `JSONField` | 模板描述，支持多语言 |
| `canvas_type` | `CharField` | 模板类型 |
| `canvas_category` | `CharField` | 模板分类 |
| `dsl` | `JSONField` | 模板 DSL |

### `user_canvas_version`

- 用途：用户画布版本快照表
- 主键：`id`
- 当前行数：`0`
- 逻辑关系：
  - `user_canvas_id` 指向 `user_canvas`

| 字段名 | 类型 | 含义 |
|---|---|---|
| `id` | `CharField` | 版本 ID |
| `user_canvas_id` | `CharField` | 所属画布 ID |
| `title` | `CharField` | 版本标题 |
| `description` | `TextField` | 版本描述 |
| `release` | `BooleanField` | 是否为发布版本 |
| `dsl` | `JSONField` | 版本 DSL 快照 |

## 6. 建议阅读顺序

如果你想从数据库角度理解整个项目，建议按这个顺序读：

1. `user`
2. `file`
3. `llm_factories`
4. `llm`
5. `knowledgebase`
6. `document`
7. `dialog`
8. `conversation`
9. `search`
10. `memory`
11. `user_canvas`

这样会比较符合项目从“账号 -> 模型 -> 知识库 -> 对话 -> 工作流”的理解路径。

## 7. 与你当前问题最相关的表

如果聚焦到“前端写死 `DIALOG_ID` 为什么不能工作”这个问题，最关键的是下面几张表：

- `dialog`
- `conversation`
- `user`
- `knowledgebase`
- `document`

其中最核心的是：

- `dialog`：必须先有一条对话应用记录
- `conversation`：只有在 `dialog` 存在后，才能正常创建会话

你当前数据库的关键事实是：

- `dialog` 目前有 `0` 行
- `conversation` 目前有 `0` 行
- `knowledgebase` 目前有 `0` 行
- `document` 目前有 `0` 行

所以现在的状态不是“数据库没初始化”，而是：

- 系统表初始化了
- 用户表初始化了
- 但聊天业务数据还没有创建

换句话说，**当前缺的不是建表，而是 `dialog` 这类业务记录。**

