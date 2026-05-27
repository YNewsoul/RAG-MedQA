"""Prompt 与引用辅助函数模块。

这是 RAG 系统中连接「检索结果」和「最终 Prompt」的关键辅助层，
负责将检索到的知识片段格式化为 LLM 可理解的上下文，并提供多种
NLP 辅助功能。

核心功能模块：
┌─────────────────────────────────────────────────────────────┐
│ 1. 检索结果处理                                             │
│    - chunks_format: 规范化不同来源的 chunk 字段               │
│    - kb_prompt: 将 chunk 拼接成知识段                       │
│    - message_fit_in: 消息上下文裁剪（控制 token 数）          │
├─────────────────────────────────────────────────────────────┤
│ 2. 引用标注系统                                             │
│    - citation_prompt: 引用格式提示词                        │
│    - citation_plus: 增强版引用提示词                        │
├─────────────────────────────────────────────────────────────┤
│ 3. NLP 辅助功能                                            │
│    - keyword_extraction: 关键词提取                         │
│    - full_question: 多轮对话整合                            │
│    - cross_languages: 跨语言转换                            │
│    - content_tagging: 内容标签标注                          │
├─────────────────────────────────────────────────────────────┤
│ 4. 工具调用与任务管理                                       │
│    - tool_schema: 工具描述格式化                            │
│    - analyze_task_async: 任务分析                           │
│    - next_step_async: 下一步计划生成                        │
│    - reflect_async: 反思总结                               │
├─────────────────────────────────────────────────────────────┤
│ 5. 多模态与文档处理                                         │
│    - vision_llm_describe_prompt: 图像描述提示词             │
│    - detect_table_of_contents: 目录检测                     │
│    - extract_table_of_contents: 目录提取                    │
└─────────────────────────────────────────────────────────────┘
"""

import asyncio
import datetime
import json
import logging
import re
from copy import deepcopy
from typing import Tuple
import jinja2
import json_repair
from common.misc_utils import hash_str2int
from rag.nlp import rag_tokenizer
from rag.prompts.template import load_prompt
from common.constants import TAG_FLD
from common.token_utils import encoder, num_tokens_from_string

# ==================== 常量定义 ====================
STOP_TOKEN = "<|STOP|>"      # LLM 生成停止标记
COMPLETE_TASK = "complete_task"  # 工具调用任务完成标记
INPUT_UTILIZATION = 0.5       # 输入 token 利用率（用于控制上下文长度）


def get_value(d, k1, k2):
    """从字典中获取值，支持两个备选键。

    用于处理不同数据源返回的字段命名不一致问题。
    
    Args:
        d (dict): 输入字典
        k1 (str): 首选键名
        k2 (str): 备选键名
        
    Returns:
        Any: 对应键的值，如果两个键都不存在返回 None
    """
    return d.get(k1, d.get(k2))


def chunks_format(reference):
    """把后端内部引用结构规范化为前端统一的 chunk 字段。

    由于不同检索源（向量数据库、SQL查询、网页检索等）返回的字段命名可能不同，
    此函数将各种字段名统一映射为标准格式，便于后续处理和展示。
    
    标准化字段映射：
    - id: chunk_id / id
    - content: content / content_with_weight
    - document_id: doc_id / document_id
    - document_name: docnm_kwd / document_name
    - dataset_id: kb_id / dataset_id
    - image_id: image_id / img_id
    - positions: positions / position_int
    
    Args:
        reference (dict): 检索结果字典，必须包含 chunks 字段
        
    Returns:
        list[dict]: 规范化后的 chunk 列表，每个 chunk 包含标准字段
    """
    if not reference or not isinstance(reference, dict):
        return []
    raw_chunks = reference.get("chunks", [])
    if not isinstance(raw_chunks, list):
        return []
    # 使用列表推导式规范化每个 chunk
    return [
        {
            # chunk 唯一标识（支持 chunk_id 和 id 两种字段名）
            "id": get_value(chunk, "chunk_id", "id"),
            # chunk 内容（优先 content，备选 content_with_weight）
            "content": get_value(chunk, "content", "content_with_weight"),
            # 所属文档 ID（支持 doc_id 和 document_id）
            "document_id": get_value(chunk, "doc_id", "document_id"),
            # 文档名称（支持 docnm_kwd 和 document_name）
            "document_name": get_value(chunk, "docnm_kwd", "document_name"),
            # 所属数据集/知识库 ID（支持 kb_id 和 dataset_id）
            "dataset_id": get_value(chunk, "kb_id", "dataset_id"),
            # 图片 ID（支持 image_id 和 img_id）
            "image_id": get_value(chunk, "image_id", "img_id"),
            # 在文档中的位置（支持 positions 和 position_int）
            "positions": get_value(chunk, "positions", "position_int"),
            # 来源 URL
            "url": chunk.get("url"),
            # 综合相似度分数
            "similarity": chunk.get("similarity"),
            # 向量相似度分数
            "vector_similarity": chunk.get("vector_similarity"),
            # 词项相似度分数
            "term_similarity": chunk.get("term_similarity"),
            # 数据库行 ID（SQL 检索结果）
            "row_id": chunk.get("row_id"),
            # 文档类型（支持 doc_type_kwd 和 doc_type）
            "doc_type": get_value(chunk, "doc_type_kwd", "doc_type"),
        }
        # 遍历所有原始 chunk
        for chunk in raw_chunks
        # 过滤掉非字典类型的无效数据
        if isinstance(chunk, dict)
    ]


def message_fit_in(msg, max_length=4000):
    """在尽量保留系统提示与最新消息的前提下裁剪上下文。

    Token 裁剪策略（优先保证关键信息）：
    1. 首先计算总 token 数，如果未超限制则直接返回
    2. 只保留系统提示和最新一条消息（丢弃历史对话）
    3. 如果仍然超限：
       - 系统提示占比 > 80% → 裁剪系统提示
       - 否则 → 裁剪最新消息

    Args:
        msg (list[dict]): 消息列表，每个消息包含 role 和 content 字段
        max_length (int): 最大 token 数限制（默认 4000）
        
    Returns:
        tuple[int, list[dict]]: (裁剪后的总 token 数, 裁剪后的消息列表)
    """
    def count(messages=None):
        """计算消息总 token 数。"""
        messages = messages if messages is not None else msg
        total = 0
        for m in messages:
            total += num_tokens_from_string(str(m.get("content", "")))
        return total

    def truncate_message_content(messages, indexes, budget):
        """按给定预算顺序裁剪若干消息的 content。"""
        remaining = max(int(budget), 0)
        for idx in indexes:
            content = str(messages[idx].get("content", ""))
            tokens = encoder.encode(content)
            kept = tokens[:remaining]
            messages[idx]["content"] = encoder.decode(kept)
            remaining -= len(kept)
        return messages

    if not msg:
        return 0, []

    # 在函数内部工作于浅拷贝消息列表，避免直接修改调用方原始对象。
    msg = [dict(m) for m in msg]

    # 第一步：检查是否超限，如果未超限则直接返回
    c = count()
    if c <= max_length:
        return c, msg

    # 第二步：首次裁剪 - 只保留系统提示和最新一条消息
    # 系统提示包含重要的指令信息，最新消息是用户当前查询，这两者是最关键的。
    system_indexes = [i for i, m in enumerate(msg) if m.get("role") == "system"]
    keep_indexes = list(system_indexes)
    latest_index = len(msg) - 1
    if not keep_indexes or keep_indexes[-1] != latest_index:
        keep_indexes.append(latest_index)
    msg = [dict(msg[i]) for i in keep_indexes]

    c = count(msg)
    if c <= max_length:
        return c, msg

    # 第三步：二次裁剪 - 需要进一步缩减内容
    system_indexes = [i for i, m in enumerate(msg) if m.get("role") == "system"]
    latest_index = len(msg) - 1 if msg else -1

    system_token_count = sum(
        num_tokens_from_string(str(msg[i].get("content", ""))) for i in system_indexes
    )
    latest_is_system = latest_index in system_indexes
    latest_token_count = 0
    if latest_index >= 0 and not latest_is_system:
        latest_token_count = num_tokens_from_string(str(msg[latest_index].get("content", "")))

    # 只有系统提示时，只能裁系统提示。
    if latest_is_system:
        msg = truncate_message_content(msg, system_indexes, max_length)
        return count(msg), msg

    # 没有系统提示时，只能裁最新消息。
    if not system_indexes:
        msg = truncate_message_content(msg, [latest_index], max_length)
        return count(msg), msg

    total_critical_tokens = system_token_count + latest_token_count
    system_ratio = system_token_count / total_critical_tokens if total_critical_tokens else 1.0

    # 如果系统提示占比超过 80%，优先裁系统提示；否则裁最新消息。
    if system_ratio > 0.8:
        msg = truncate_message_content(msg, system_indexes, max_length - latest_token_count)
    else:
        msg = truncate_message_content(msg, [latest_index], max_length - system_token_count)

    return count(msg), msg


def kb_prompt(kbinfos, max_tokens, hash_id=False):
    """把召回的 chunk 列表拼接成注入模型的知识段（Knowledge Prompt）。

    核心功能：
    1. 根据 token 限制筛选 chunk（保留前 N 个不超过 97% max_tokens 的 chunk）
    2. 获取每个 chunk 对应的文档元数据
    3. 格式化每个 chunk，添加 ID、标题、URL、元数据和内容
    4. 返回格式化后的知识片段列表

    输出格式示例：
    ```
    ID: 0
    ├── Title: 医学指南.pdf
    ├── URL: http://example.com
    ├── Author: Dr. Smith
    └── Content:
        这是 chunk 的内容...
    ```

    Args:
        kbinfos (dict): 检索结果，包含 chunks 和 doc_aggs 字段
        max_tokens (int): 最大 token 数限制
        hash_id (bool): 是否对 chunk ID 进行哈希处理（用于隐私保护）

    Returns:
        list[str]: 格式化后的知识片段列表
    """
    # 延迟导入，避免循环依赖
    from api.db.services.document_service import DocumentService
    from api.db.services.doc_metadata_service import DocMetadataService

    # ========== 阶段1: 提取 chunk 内容并按 token 限制筛选 ==========
    # 从检索结果中提取所有 chunk 的内容
    knowledges = [get_value(ck, "content", "content_with_weight") for ck in kbinfos["chunks"]]
    kwlg_len = len(knowledges)  # 原始 chunk 总数
    used_token_count = 0        # 已使用的 token 数
    chunks_num = 0              # 最终保留的 chunk 数量

    # 遍历所有 chunk，累加 token 直到接近限制
    for i, c in enumerate(knowledges):
        if not c:  # 跳过空内容
            continue
        used_token_count += num_tokens_from_string(c)
        chunks_num += 1
        # 超过 97% 的最大 token 限制时停止（预留 3% 缓冲）
        if max_tokens * 0.97 < used_token_count:
            knowledges = knowledges[:i]  # 截断到当前位置（不含当前 chunk）
            logging.warning(f"Not all the retrieval into prompt: {len(knowledges)}/{kwlg_len}")
            break

    # ========== 阶段2: 获取文档元数据 ==========
    # 获取保留的 chunk 所属的文档信息
    doc_ids = [get_value(ck, "doc_id", "document_id") for ck in kbinfos["chunks"][:chunks_num]]
    docs = DocumentService.get_by_ids(doc_ids)

    # 为每个文档获取元数据（如作者、日期等）
    docs_with_meta = {}
    for d in docs:
        meta = DocMetadataService.get_document_metadata(d.id)
        docs_with_meta[d.id] = meta if meta else {}
    docs = docs_with_meta  # 转换为 {doc_id: metadata} 的字典格式

    # ========== 阶段3: 定义格式化辅助函数 ==========
    def draw_node(k, line):
        """格式化单个元数据节点。

        Args:
            k (str): 元数据键名
            line: 元数据值

        Returns:
            str: 格式化后的节点字符串，如 "\n├── Author: Dr. Smith"
        """
        # 确保值是字符串类型
        if line is not None and not isinstance(line, str):
            line = str(line)
        if not line:  # 空值不输出
            return ""
        # 将多行内容转为单行（替换换行符为空格）
        return f"\n├── {k}: " + re.sub(r"\n+", " ", line, flags=re.DOTALL)

    # ========== 阶段4: 构建最终的知识片段列表 ==========
    knowledges = []
    for i, ck in enumerate(kbinfos["chunks"][:chunks_num]):
        # 生成 chunk ID（可选哈希处理）
        chunk_id = i if not hash_id else hash_str2int(get_value(ck, "id", "chunk_id"), 500)
        cnt = "\nID: {}".format(chunk_id)

        # 添加文档标题
        cnt += draw_node("Title", get_value(ck, "docnm_kwd", "document_name"))

        # 添加 URL（如果存在）
        cnt += draw_node("URL", ck['url']) if "url" in ck else ""

        # 添加文档元数据（如作者、日期等）
        doc_id = get_value(ck, "doc_id", "document_id")
        for k, v in docs.get(doc_id, {}).items():
            cnt += draw_node(k, v)

        # 添加内容部分（使用 └── 表示最后一个节点）
        cnt += "\n└── Content:\n"
        cnt += get_value(ck, "content", "content_with_weight")

        knowledges.append(cnt)

    return knowledges


def memory_prompt(message_list, max_tokens):
    """从消息列表中提取内容，按 token 限制截断。

    用于构建记忆模块的提示词，按顺序累加消息内容直到达到 token 限制。

    Args:
        message_list (list[dict]): 消息列表，每个消息包含 content 字段
        max_tokens (int): 最大 token 数限制

    Returns:
        list[str]: 提取的内容列表
    """
    used_token_count = 0
    content_list = []
    for message in message_list:
        current_content_tokens = num_tokens_from_string(message["content"])
        if used_token_count + current_content_tokens > max_tokens * 0.97:
            logging.warning(f"Not all messages fit into prompt: {len(content_list)}/{len(message_list)}")
            break
        content_list.append(message["content"])
        used_token_count += current_content_tokens
    return content_list


CITATION_PROMPT_TEMPLATE = load_prompt("citation_prompt")
CITATION_PLUS_TEMPLATE = load_prompt("citation_plus")
CONTENT_TAGGING_PROMPT_TEMPLATE = load_prompt("content_tagging_prompt")
CROSS_LANGUAGES_SYS_PROMPT_TEMPLATE = load_prompt("cross_languages_sys_prompt")
CROSS_LANGUAGES_USER_PROMPT_TEMPLATE = load_prompt("cross_languages_user_prompt")
FULL_QUESTION_PROMPT_TEMPLATE = load_prompt("full_question_prompt")
KEYWORD_PROMPT_TEMPLATE = load_prompt("keyword_prompt")
QUESTION_PROMPT_TEMPLATE = load_prompt("question_prompt")
VISION_LLM_DESCRIBE_PROMPT = load_prompt("vision_llm_describe_prompt")
VISION_LLM_FIGURE_DESCRIBE_PROMPT = load_prompt("vision_llm_figure_describe_prompt")
# ========== 多模态提示词模板 ==========
VISION_LLM_FIGURE_DESCRIBE_PROMPT_WITH_CONTEXT = load_prompt("vision_llm_figure_describe_prompt_with_context")
STRUCTURED_OUTPUT_PROMPT = load_prompt("structured_output_prompt")

# ========== 任务分析与反思提示词模板 ==========
ANALYZE_TASK_SYSTEM = load_prompt("analyze_task_system")      # 任务分析系统提示
ANALYZE_TASK_USER = load_prompt("analyze_task_user")          # 任务分析用户提示
NEXT_STEP = load_prompt("next_step")                          # 下一步计划生成
REFLECT = load_prompt("reflect")                              # 反思总结
SUMMARY4MEMORY = load_prompt("summary4memory")                # 记忆摘要
RANK_MEMORY = load_prompt("rank_memory")                      # 记忆排名
META_FILTER = load_prompt("meta_filter")                      # 元数据过滤器生成
ASK_SUMMARY = load_prompt("ask_summary")                      # 摘要询问

# ========== Jinja2 模板环境配置 ==========
# autoescape=False: 不自动转义 HTML（因为我们处理的是文本提示词）
# trim_blocks=True: 移除块标签周围的空白行
# lstrip_blocks=True: 移除块标签首行的缩进
PROMPT_JINJA_ENV = jinja2.Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)


def citation_prompt(user_defined_prompts: dict = {}) -> str:
    """生成引用格式指导提示词。

    用于指导 LLM 在回答时正确引用来源文档。
    支持用户自定义的引用格式规范。

    Args:
        user_defined_prompts (dict): 用户自定义提示词字典

    Returns:
        str: 渲染后的引用指导提示词
    """
    # 优先使用用户自定义的引用指导，否则使用默认模板
    template = PROMPT_JINJA_ENV.from_string(
        user_defined_prompts.get("citation_guidelines", CITATION_PROMPT_TEMPLATE)
    )
    return template.render()


def citation_plus(sources: str) -> str:
    """生成增强版引用提示词（包含示例）。

    将引用格式示例和实际来源列表组合，生成更完整的引用指导。

    Args:
        sources (str): 来源列表字符串

    Returns:
        str: 增强版引用提示词
    """
    template = PROMPT_JINJA_ENV.from_string(CITATION_PLUS_TEMPLATE)
    # 将默认引用格式示例和来源列表传递给模板
    return template.render(example=citation_prompt(), sources=sources)


async def keyword_extraction(chat_mdl, content, topn=3):
    """从内容中提取关键词，用于增强检索准确性。

    使用 LLM 从输入内容中提取 topn 个最相关的关键词，
    这些关键词会被追加到用户问题中，提升向量检索的匹配效果。

    Args:
        chat_mdl: LLM 模型实例
        content (str): 输入内容（用户问题）
        topn (int): 提取关键词数量（默认 3）

    Returns:
        str: 提取的关键词字符串（格式如 "[关键词1, 关键词2, 关键词3]"）
    """
    template = PROMPT_JINJA_ENV.from_string(KEYWORD_PROMPT_TEMPLATE)
    rendered_prompt = template.render(content=content, topn=topn)

    msg = [{"role": "system", "content": rendered_prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = await chat_mdl.async_chat(rendered_prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


async def question_proposal(chat_mdl, content, topn=3):
    """根据内容生成相关问题建议。

    使用 LLM 从输入内容中生成 topn 个相关问题，
    可用于引导用户进行更深入的提问或扩展对话。

    Args:
        chat_mdl: LLM 模型实例
        content (str): 输入内容
        topn (int): 生成问题数量（默认 3）

    Returns:
        str: 生成的问题字符串
    """
    template = PROMPT_JINJA_ENV.from_string(QUESTION_PROMPT_TEMPLATE)
    rendered_prompt = template.render(content=content, topn=topn)

    msg = [{"role": "system", "content": rendered_prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = await chat_mdl.async_chat(rendered_prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


async def full_question(llm_id=None, messages=[], language=None, chat_mdl=None):
    """将多轮对话历史整合为单个完整问题。

    在多轮对话场景中，用户的问题可能依赖上下文，此函数通过 LLM
    将历史对话压缩成一个独立的完整问题，便于检索和理解。

    功能特点：
    - 自动获取或创建 LLM 实例
    - 过滤非用户/助手消息
    - 生成包含日期信息（今日/昨日/明日）的提示词
    - 失败时回退到原始最后一条消息

    Args:
        llm_id (str): LLM 模型 ID（可选）
        messages (list[dict]): 对话消息列表
        language (str): 目标语言（可选）
        chat_mdl: 预创建的 LLM 模型实例（可选）

    Returns:
        str: 整合后的完整问题
    """
    from common.constants import LLMType
    from api.db.services.llm_service import LLMBundle
    from api.db.services.tenant_llm_service import TenantLLMService
    from api.db.joint_services.tenant_model_service import get_model_config_by_type_and_name

    # ========== 步骤1: 获取或创建 LLM 实例 ==========
    # 如果没有提供模型实例，根据 llm_id 创建
    if not chat_mdl:
        # 判断模型类型（image2text 或 chat）
        if TenantLLMService.llm_id2llm_type(llm_id) == "image2text":
            chat_model_config = get_model_config_by_type_and_name(LLMType.IMAGE2TEXT, llm_id)
        else:
            chat_model_config = get_model_config_by_type_and_name(LLMType.CHAT, llm_id)
        # 创建 LLMBundle 实例
        chat_mdl = LLMBundle(chat_model_config)

    # ========== 步骤2: 构建对话历史字符串 ==========
    # 过滤出用户和助手的消息（忽略系统消息等）
    conv = []
    for m in messages:
        if m["role"] not in ["user", "assistant"]:
            continue
        # 格式化为 "USER: ..." 或 "ASSISTANT: ..."
        conv.append("{}: {}".format(m["role"].upper(), m["content"]))
    # 用换行符连接所有消息
    conversation = "\n".join(conv)

    # ========== 步骤3: 获取日期信息 ==========
    # 用于解析对话中的相对时间（如 "昨天"、"明天"）
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    # ========== 步骤4: 渲染提示词模板 ==========
    template = PROMPT_JINJA_ENV.from_string(FULL_QUESTION_PROMPT_TEMPLATE)
    rendered_prompt = template.render(
        today=today,
        yesterday=yesterday,
        tomorrow=tomorrow,
        conversation=conversation,
        language=language,
    )

    # ========== 步骤5: 调用 LLM 生成完整问题 ==========
    ans = await chat_mdl.async_chat(rendered_prompt, [{"role": "user", "content": "Output: "}])
    # 移除可能的思考标签
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)

    # ========== 步骤6: 错误处理与回退 ==========
    # 如果生成失败（包含错误标记），回退到原始最后一条消息
    if ans.find("**ERROR**") >= 0:
        return messages[-1]["content"] if messages else ""
    return ans


async def cross_languages(llm_id, query, languages=[]):
    """将问题转换为目标语言。

    在多语言场景中，用户可能用非知识库语言提问，此函数将问题
    转换为知识库的目标语言，提升检索准确性。

    功能特点：
    - 支持多种目标语言
    - 失败时回退到原始查询
    - 自动处理 image2text 和 chat 类型模型

    Args:
        llm_id (str): LLM 模型 ID
        query (str): 用户查询
        languages (list): 目标语言列表

    Returns:
        str: 转换后的查询（可能包含多种语言版本）
    """
    from common.constants import LLMType
    from api.db.services.llm_service import LLMBundle
    from api.db.services.tenant_llm_service import TenantLLMService
    from api.db.joint_services.tenant_model_service import get_model_config_by_type_and_name, get_tenant_default_model_by_type

    # ========== 步骤1: 获取模型配置 ==========
    # 判断模型类型并获取配置
    if llm_id and TenantLLMService.llm_id2llm_type(llm_id) == "image2text":
        # 如果是图像转文本模型
        chat_model_config = get_model_config_by_type_and_name(LLMType.IMAGE2TEXT, llm_id)
    else:
        # 如果是聊天模型或没有指定 llm_id
        if not llm_id:
            # 使用租户默认的聊天模型
            chat_model_config = get_tenant_default_model_by_type(LLMType.CHAT)
        else:
            # 根据 llm_id 获取聊天模型配置
            chat_model_config = get_model_config_by_type_and_name(LLMType.CHAT, llm_id)

    # 创建 LLM 实例
    chat_mdl = LLMBundle(chat_model_config)

    # ========== 步骤2: 渲染提示词 ==========
    # 系统提示：定义翻译任务角色
    rendered_sys_prompt = PROMPT_JINJA_ENV.from_string(CROSS_LANGUAGES_SYS_PROMPT_TEMPLATE).render()
    # 用户提示：包含查询内容和目标语言列表
    rendered_user_prompt = PROMPT_JINJA_ENV.from_string(CROSS_LANGUAGES_USER_PROMPT_TEMPLATE).render(
        query=query,
        languages=languages
    )

    # ========== 步骤3: 调用 LLM 进行翻译 ==========
    ans = await chat_mdl.async_chat(
        rendered_sys_prompt, 
        [{"role": "user", "content": rendered_user_prompt}],
        {"temperature": 0.2}  # 低温度确保翻译准确性
    )
    # 移除思考标签
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)

    # ========== 步骤4: 错误处理与结果解析 ==========
    # 如果翻译失败，回退到原始查询
    if ans.find("**ERROR**") >= 0:
        return query

    # 解析多语言输出（LLM 返回用 === 分隔的多种语言版本）
    # 1. 移除 "Output:" 前缀和多余换行
    # 2. 用 === 分割
    # 3. 过滤空字符串
    cleaned_ans = re.sub(r"(^Output:|\n+)", "", ans, flags=re.DOTALL)
    language_versions = [a for a in cleaned_ans.split("===") if a.strip()]
    return "\n".join(language_versions)


async def content_tagging(chat_mdl, content, all_tags, examples, topn=3):
    """为内容添加标签（零样本/少样本分类）。

    使用 LLM 对内容进行标签标注，支持零样本和少样本学习模式。
    返回一个字典，包含标签及其置信度评分。

    Args:
        chat_mdl: LLM 模型实例
        content (str): 待标注内容
        all_tags (list): 所有可用标签列表
        examples (list): 示例数据（用于少样本学习）
        topn (int): 最多返回标签数（默认 3）

    Returns:
        dict: {标签名: 置信度} 的字典

    Raises:
        Exception: 标注失败时抛出异常
    """
    template = PROMPT_JINJA_ENV.from_string(CONTENT_TAGGING_PROMPT_TEMPLATE)

    # 预处理示例数据
    for ex in examples:
        ex["tags_json"] = json.dumps(ex[TAG_FLD], indent=2, ensure_ascii=False)

    # 渲染提示词
    rendered_prompt = template.render(
        topn=topn,
        all_tags=all_tags,
        examples=examples,
        content=content,
    )

    # 调用 LLM
    msg = [{"role": "system", "content": rendered_prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = await chat_mdl.async_chat(rendered_prompt, msg[1:], {"temperature": 0.5})
    # 处理返回结果（如果是元组，取第一个元素）
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    # 移除思考标签
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)

    # ========== 错误处理 ==========
    # 如果 LLM 返回错误标记，抛出异常
    if kwd.find("**ERROR**") >= 0:
        raise Exception(kwd)

    # ========== 解析 JSON 输出（带容错处理） ==========
    try:
        # 首先尝试直接解析
        obj = json_repair.loads(kwd)
    except json_repair.JSONDecodeError:
        # 容错处理：LLM 可能返回不规范的 JSON
        try:
            # 1. 移除可能混入的提示词内容
            result = kwd.replace(rendered_prompt[:-1], "").replace("user", "").replace("model", "").strip()
            # 2. 提取最外层的 JSON 对象（取第一个 { 和最后一个 } 之间的内容）
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            # 3. 再次尝试解析
            obj = json_repair.loads(result)
        except Exception as e:
            logging.exception(f"JSON parsing error: {result} -> {e}")
            raise e

    # ========== 过滤有效标签 ==========
    # 只保留置信度 > 0 的标签
    res = {}
    for k, v in obj.items():
        try:
            # 确保值是整数且大于 0
            if int(v) > 0:
                res[str(k)] = int(v)
        except Exception:
            # 跳过无效的标签值
            pass
    return res


def vision_llm_describe_prompt(page=None) -> str:
    """生成视觉 LLM 的图像描述提示词（基础版本）。

    用于指导视觉 LLM 对图像进行描述。

    Args:
        page: 页面信息（可选）

    Returns:
        str: 渲染后的图像描述提示词
    """
    template = PROMPT_JINJA_ENV.from_string(VISION_LLM_DESCRIBE_PROMPT)
    return template.render(page=page)


def vision_llm_figure_describe_prompt() -> str:
    """生成视觉 LLM 的图表描述提示词。

    专门用于描述文档中的图表（如图表、图形、表格等）。

    Returns:
        str: 渲染后的图表描述提示词
    """
    template = PROMPT_JINJA_ENV.from_string(VISION_LLM_FIGURE_DESCRIBE_PROMPT)
    return template.render()


def vision_llm_figure_describe_prompt_with_context(context_above: str, context_below: str) -> str:
    """生成带上下文的视觉 LLM 图表描述提示词。

    在描述图表时，结合图表上方和下方的文本上下文，
    可以更准确地理解图表的含义。

    Args:
        context_above (str): 图表上方的文本上下文
        context_below (str): 图表下方的文本上下文

    Returns:
        str: 渲染后的带上下文图表描述提示词
    """
    template = PROMPT_JINJA_ENV.from_string(VISION_LLM_FIGURE_DESCRIBE_PROMPT_WITH_CONTEXT)
    return template.render(context_above=context_above, context_below=context_below)


def tool_schema(tools_description: list[dict], complete_task=False):
    """将工具描述列表格式化为 LLM 可理解的工具说明字符串。

    将工具列表转换为带编号的 Markdown 格式，便于 LLM 理解和调用。
    可选添加 "complete_task" 工具，用于标记任务完成。

    Args:
        tools_description (list[dict]): 工具描述列表，每个工具包含 function 字段
        complete_task (bool): 是否添加任务完成工具（默认 False）

    Returns:
        str: 格式化后的工具说明字符串
    """
    if not tools_description:
        return ""

    desc = {}

    # 添加任务完成工具（如果需要）
    if complete_task:
        desc[COMPLETE_TASK] = {
            "type": "function",
            "function": {
                "name": COMPLETE_TASK,
                "description": "When you have the final answer and are ready to complete the task, call this function with your answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "The final answer to the user's question"}},
                    "required": ["answer"]
                }
            }
        }

    # 添加自定义工具
    for idx, tool in enumerate(tools_description):
        name = tool["function"]["name"]
        desc[name] = tool

    # 格式化为带编号的 Markdown
    return "\n\n".join([
        f"## {i + 1}. {fnm}\n{json.dumps(des, ensure_ascii=False, indent=4)}"
        for i, (fnm, des) in enumerate(desc.items())
    ])


def form_history(history, limit=-6):
    """将对话历史格式化为上下文字符串。

    从对话历史中提取最近的消息（默认最近6条），格式化为 USER/AGENT 标记的上下文。
    每条消息内容最多保留 2048 字符。

    Args:
        history (list[dict]): 对话历史列表
        limit (int): 保留消息数量（负数表示倒数）

    Returns:
        str: 格式化后的上下文字符串
    """
    context = ""
    for h in history[limit:]:
        if h["role"] == "system":
            continue
        role = "USER"
        if h["role"].upper() != role:
            role = "AGENT"
        content = h["content"][:2048]
        if len(h["content"]) > 2048:
            content += "..."
        context += f"\n{role}: {content}"
    return context


async def analyze_task_async(chat_mdl, prompt, task_name, tools_description: list[dict],
                             user_defined_prompts: dict = {}):
    """分析任务需求，生成任务执行计划。

    使用 LLM 分析用户的任务需求，结合可用工具，生成任务执行计划。
    支持用户自定义的任务分析提示词。

    Args:
        chat_mdl: LLM 模型实例
        prompt (str): Agent 系统提示词
        task_name (str): 任务名称
        tools_description (list[dict]): 可用工具列表
        user_defined_prompts (dict): 用户自定义提示词（可选）

    Returns:
        str: 任务分析结果
    """
    # ========== 步骤1: 格式化工具描述 ==========
    tools_desc = tool_schema(tools_description)
    context = ""

    # ========== 步骤2: 选择并渲染提示词模板 ==========
    # 优先使用用户自定义的任务分析提示词
    if user_defined_prompts.get("task_analysis"):
        template = PROMPT_JINJA_ENV.from_string(user_defined_prompts["task_analysis"])
    else:
        # 使用默认的任务分析提示词（系统提示 + 用户提示）
        template = PROMPT_JINJA_ENV.from_string(ANALYZE_TASK_SYSTEM + "\n\n" + ANALYZE_TASK_USER)

    # 渲染提示词模板
    context = template.render(
        task=task_name, 
        context=context, 
        agent_prompt=prompt, 
        tools_desc=tools_desc
    )

    # ========== 步骤3: 调用 LLM 进行任务分析 ==========
    kwd = await chat_mdl.async_chat(
        context, 
        [{"role": "user", "content": "Please analyze it."}]
    )

    # 处理返回结果
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)

    # 错误处理
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


async def next_step_async(chat_mdl, history: list, tools_description: list[dict], task_desc,
                          user_defined_prompts: dict = {}):
    """根据对话历史和任务描述，决定下一步要调用的工具。

    智能体核心函数之一，根据当前对话状态决定：
    - 调用哪个工具
    - 是否完成任务（调用 complete_task）

    Args:
        chat_mdl: LLM 模型实例
        history (list[dict]): 对话历史
        tools_description (list[dict]): 可用工具列表
        task_desc (str): 任务描述
        user_defined_prompts (dict): 用户自定义提示词（可选）

    Returns:
        tuple[str, int]: (下一步工具调用的 JSON 字符串, token 数量)
    """
    # ========== 边界检查 ==========
    if not tools_description:
        return "", 0

    # ========== 步骤1: 格式化工具描述 ==========
    desc = tool_schema(tools_description)

    # ========== 步骤2: 获取提示词模板 ==========
    # 优先使用用户自定义的计划生成提示词
    template = PROMPT_JINJA_ENV.from_string(
        user_defined_prompts.get("plan_generation", NEXT_STEP)
    )

    # ========== 步骤3: 构造用户提示 ==========
    user_prompt = "\nWhat's the next tool to call? If ready OR IMPOSSIBLE TO BE READY, then call `complete_task`."

    # 深拷贝历史记录以避免修改原数据
    hist = deepcopy(history)
    # 将用户提示追加到最后一条消息
    if hist[-1]["role"] == "user":
        hist[-1]["content"] += user_prompt
    else:
        hist.append({"role": "user", "content": user_prompt})

    # ========== 步骤4: 调用 LLM 生成下一步计划 ==========
    json_str = await chat_mdl.async_chat(
        template.render(
            task_analysis=task_desc,  # 任务分析结果
            desc=desc,                 # 工具描述
            today=datetime.datetime.now().strftime("%Y-%m-%d")  # 当前日期
        ),
        hist[1:],  # 跳过第一条消息（通常是系统提示）
        stop=["<|stop|>"],  # 指定停止标记
    )

    # ========== 步骤5: 处理结果 ==========
    tk_cnt = num_tokens_from_string(json_str)  # 计算 token 数
    json_str = re.sub(r"^.*</think>", "", json_str, flags=re.DOTALL)  # 移除思考标签

    return json_str, tk_cnt


async def reflect_async(chat_mdl, history: list[dict], tool_call_res: list[Tuple], user_defined_prompts: dict = {}):
    """对工具调用结果进行反思总结。

    智能体反思机制，分析已执行的工具调用，评估是否达成目标，
    为下一步决策提供依据。

    Args:
        chat_mdl: LLM 模型实例
        history (list[dict]): 对话历史
        tool_call_res (list[Tuple]): 工具调用结果列表 (工具名, 结果)
        user_defined_prompts (dict): 用户自定义提示词（可选）

    Returns:
        str: 反思总结字符串（包含 Observation 和 Reflection 两部分）
    """
    # 格式化工具调用结果
    tool_calls = [{"name": p[0], "result": p[1]} for p in tool_call_res]
    # 获取原始目标（历史记录的第二条消息通常是用户的初始问题）
    goal = history[1]["content"]

    # ========== 步骤2: 渲染反思提示词 ==========
    # 优先使用用户自定义的反思提示词
    template = PROMPT_JINJA_ENV.from_string(user_defined_prompts.get("reflection", REFLECT))
    user_prompt = template.render(goal=goal, tool_calls=tool_calls)

    # ========== 步骤3: 构造对话历史 ==========
    hist = deepcopy(history)
    if hist[-1]["role"] == "user":
        hist[-1]["content"] += user_prompt
    else:
        hist.append({"role": "user", "content": user_prompt})

    # ========== 步骤4: 裁剪消息并调用 LLM ==========
    _, msg = message_fit_in(hist, chat_mdl.max_length)  # 确保不超过 token 限制
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:])
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)  # 移除思考标签

    # ========== 步骤5: 格式化反思结果 ==========
    # 返回包含 Observation（观察）和 Reflection（反思）两部分的结果
    return """
**Observation**
{}

**Reflection**
{}
    """.format(json.dumps(tool_calls, ensure_ascii=False, indent=2), ans)


def form_message(system_prompt, user_prompt):
    """将系统提示和用户提示组合成标准消息格式。

    Args:
        system_prompt (str): 系统提示词
        user_prompt (str): 用户提示词

    Returns:
        list[dict]: 标准格式的消息列表
    """
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def structured_output_prompt(schema=None) -> str:
    """生成结构化输出提示词（用于引导 LLM 输出特定格式）。

    Args:
        schema: JSON Schema 定义（可选）

    Returns:
        str: 渲染后的结构化输出提示词
    """
    template = PROMPT_JINJA_ENV.from_string(STRUCTURED_OUTPUT_PROMPT)
    return template.render(schema=schema)


async def tool_call_summary(chat_mdl, name: str, params: dict, result: str, user_defined_prompts: dict = {}) -> str:
    """总结工具调用结果（用于记忆存储）。

    将工具调用的名称、参数和结果总结为简洁的文本，便于后续检索和理解。

    Args:
        chat_mdl: LLM 模型实例
        name (str): 工具名称
        params (dict): 工具调用参数
        result (str): 工具返回结果
        user_defined_prompts (dict): 用户自定义提示词（可选）

    Returns:
        str: 总结后的文本
    """
    template = PROMPT_JINJA_ENV.from_string(SUMMARY4MEMORY)
    system_prompt = template.render(
        name=name,
        params=json.dumps(params, ensure_ascii=False, indent=2),
        result=result
    )
    user_prompt = "→ Summary: "
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:])
    return re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)


async def rank_memories_async(chat_mdl, goal: str, sub_goal: str, tool_call_summaries: list[str],
                              user_defined_prompts: dict = {}):
    """对工具调用总结进行相关性排名。

    根据当前目标和子目标，对多个工具调用总结进行排名，
    确定哪些结果与当前任务最相关。

    Args:
        chat_mdl: LLM 模型实例
        goal (str): 当前目标
        sub_goal (str): 当前子目标
        tool_call_summaries (list[str]): 工具调用总结列表
        user_defined_prompts (dict): 用户自定义提示词（可选）

    Returns:
        str: 排名结果
    """
    template = PROMPT_JINJA_ENV.from_string(RANK_MEMORY)
    system_prompt = template.render(
        goal=goal, 
        sub_goal=sub_goal,
        results=[{"i": i, "content": s} for i, s in enumerate(tool_call_summaries)]
    )
    user_prompt = " → rank: "
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:], stop="<|stop|>")
    return re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)


async def gen_meta_filter(chat_mdl, meta_data: dict, query: str, constraints: dict = None) -> dict:
    meta_data_structure = {}
    for key, values in meta_data.items():
        meta_data_structure[key] = list(values.keys()) if isinstance(values, dict) else values

    sys_prompt = PROMPT_JINJA_ENV.from_string(META_FILTER).render(
        current_date=datetime.datetime.today().strftime('%Y-%m-%d'),
        metadata_keys=json.dumps(meta_data_structure),
        user_question=query,
        constraints=json.dumps(constraints) if constraints else None
    )
    user_prompt = "Generate filters:"
    ans = await chat_mdl.async_chat(sys_prompt, [{"role": "user", "content": user_prompt}])
    ans = re.sub(r"(^.*</think>|```json\n|```\n*$)", "", ans, flags=re.DOTALL)
    try:
        ans = json_repair.loads(ans)
        assert isinstance(ans, dict), ans
        assert "conditions" in ans and isinstance(ans["conditions"], list), ans
        return ans
    except Exception:
        logging.exception(f"Loading json failure: {ans}")

    return {"conditions": []}


async def gen_json(system_prompt: str, user_prompt: str, chat_mdl, gen_conf={}, max_retry=2):
    """调用 LLM 生成 JSON 格式输出（带缓存和重试机制）。

    封装了 JSON 生成的通用逻辑，包括：
    - LLM 调用结果缓存
    - 自动重试（最多 max_retry 次）
    - JSON 解析和修复
    - 错误反馈（将解析错误反馈给 LLM 进行修正）

    Args:
        system_prompt (str): 系统提示词
        user_prompt (str): 用户提示词
        chat_mdl: LLM 模型实例
        gen_conf (dict): 生成配置（如 temperature, max_tokens 等）
        max_retry (int): 最大重试次数（默认 2）

    Returns:
        dict/list: 解析后的 JSON 对象
    """
    from rag.graphrag.utils import get_llm_cache, set_llm_cache

    # ========== 步骤1: 检查缓存 ==========
    # 尝试从缓存获取结果，避免重复调用 LLM
    cached = get_llm_cache(chat_mdl.llm_name, system_prompt, user_prompt, gen_conf)
    if cached:
        return json_repair.loads(cached)

    # ========== 步骤2: 准备消息 ==========
    # 将提示词组合成标准消息格式并裁剪
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)

    err = ""
    ans = ""

    # ========== 步骤3: 带重试的 JSON 生成 ==========
    for _ in range(max_retry):
        # 如果上一次生成失败，将错误信息反馈给 LLM
        if ans and err:
            msg[-1]["content"] += f"\nGenerated JSON is as following:\n{ans}\nBut exception while loading:\n{err}\nPlease reconsider and correct it."

        # 调用 LLM
        ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:], gen_conf=gen_conf)

        # 清理输出（移除思考标签和代码块标记）
        ans = re.sub(r"(^.*</think>|```json\n|```\n*$)", "", ans, flags=re.DOTALL)

        # 尝试解析 JSON
        try:
            res = json_repair.loads(ans)
            # 缓存结果
            set_llm_cache(chat_mdl.llm_name, system_prompt, ans, user_prompt, gen_conf)
            return res
        except Exception as e:
            logging.exception(f"Loading json failure: {ans}")
            err += str(e)


TOC_DETECTION = load_prompt("toc_detection")


async def detect_table_of_contents(page_1024: list[str], chat_mdl):
    """检测文档中的目录（Table of Contents）部分。

    逐页检测文档，判断是否存在目录，直到遇到非目录页为止。
    最多检查前 22 页。

    Args:
        page_1024 (list[str]): 文档页面列表（每页约 1024 字符）
        chat_mdl: LLM 模型实例

    Returns:
        list[str]: 包含目录的页面列表
    """
    toc_secs = []
    for i, sec in enumerate(page_1024[:22]):
        ans = await gen_json(
            PROMPT_JINJA_ENV.from_string(TOC_DETECTION).render(page_txt=sec),
            "Only JSON please.",
            chat_mdl
        )
        # 如果已经找到目录且当前页不是目录，则停止
        if toc_secs and not ans["exists"]:
            break
        toc_secs.append(sec)
    return toc_secs


TOC_EXTRACTION = load_prompt("toc_extraction")
TOC_EXTRACTION_CONTINUE = load_prompt("toc_extraction_continue")


async def extract_table_of_contents(toc_pages, chat_mdl):
    """从目录页面中提取结构化的目录数据。

    使用 LLM 解析目录页面，提取章节结构、标题和页码信息。

    Args:
        toc_pages (list[str]): 包含目录的页面列表
        chat_mdl: LLM 模型实例

    Returns:
        list[dict]: 结构化的目录数据，包含 structure、title、page 等字段
    """
    if not toc_pages:
        return []

    return await gen_json(
        PROMPT_JINJA_ENV.from_string(TOC_EXTRACTION).render(toc_page="\n".join(toc_pages)),
        "Only JSON please.", 
        chat_mdl
    )


async def toc_index_extractor(toc: list[dict], content: str, chat_mdl):
    """为目录条目添加物理页码索引。

    根据文档内容，为目录中的每个章节添加对应的物理页码标记（如 <physical_index_1>）。
    这用于建立目录条目和实际文档内容之间的映射关系。

    Args:
        toc (list[dict]): 目录结构（包含 structure 和 title 字段）
        content (str): 文档页面内容（包含 <physical_index_X> 标记）
        chat_mdl: LLM 模型实例

    Returns:
        list[dict]: 包含 physical_index 的目录结构
    """
    tob_extractor_prompt = """
    You are given a table of contents in a json format and several pages of a document, 
    your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> 
    to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the 
    hierarchy section in the table of contents. For example, the first section has 
    structure index 1, the first subsection has structure index 1.1, etc.

    The response should be in the following JSON format:
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the title of the section are not in the provided pages, do not add the 
    physical_index to it.
    Directly return the final JSON structure. Do not output anything else.
    """

    prompt = tob_extractor_prompt + '\nTable of contents:\n' + json.dumps(toc, ensure_ascii=False, indent=2) + '\nDocument pages:\n' + content
    return await gen_json(prompt, "Only JSON please.", chat_mdl)


TOC_INDEX = load_prompt("toc_index")


async def table_of_contents_index(toc_arr: list[dict], sections: list[str], chat_mdl):
    """建立目录条目与文档章节的索引映射。

    该函数将目录条目与文档中的实际章节进行匹配，为每个目录条目
    添加对应的章节索引。使用深度优先搜索寻找最优匹配路径。

    Args:
        toc_arr (list[dict]): 目录数组，每个元素包含 structure 和 title 字段
        sections (list[str]): 文档章节列表
        chat_mdl: LLM 模型实例

    Returns:
        list[dict]: 更新后的目录数组，包含 indices 字段
    """
    # 边界检查
    if not toc_arr or not sections:
        return []

    # ========== 步骤1: 构建目录映射表 ==========
    # 创建两种键：带结构的标题和纯标题
    toc_map = {}
    for i, it in enumerate(toc_arr):
        k1 = (it["structure"] + it["title"]).replace(" ", "")  # 结构+标题（去空格）
        k2 = it["title"].strip()  # 纯标题
        if k1 not in toc_map:
            toc_map[k1] = []
        if k2 not in toc_map:
            toc_map[k2] = []
        toc_map[k1].append(i)
        toc_map[k2].append(i)

    # ========== 步骤2: 初步匹配 ==========
    # 为每个目录条目初始化 indices 字段
    for it in toc_arr:
        it["indices"] = []
    # 根据章节名称匹配目录条目
    for i, sec in enumerate(sections):
        sec = sec.strip()
        if sec.replace(" ", "") in toc_map:
            for j in toc_map[sec.replace(" ", "")]:
                toc_arr[j]["indices"].append(i)

    # ========== 步骤3: 使用 DFS 寻找最优匹配路径 ==========
    all_pathes = []

    def dfs(start, path):
        """深度优先搜索寻找最长有效匹配路径。

        Args:
            start (int): 当前处理的目录索引
            path (list): 当前路径（包含 (章节索引, 目录索引) 元组）
        """
        nonlocal all_pathes
        if start >= len(toc_arr):
            if path:
                all_pathes.append(path)
            return
        # 如果当前目录条目没有匹配的章节，跳过
        if not toc_arr[start]["indices"]:
            dfs(start + 1, path)
            return
        added = False
        # 尝试每个可能的章节索引
        for j in toc_arr[start]["indices"]:
            # 确保章节索引递增（保持顺序）
            if path and j < path[-1][0]:
                continue
            _path = deepcopy(path)
            _path.append((j, start))
            added = True
            dfs(start + 1, _path)
        # 如果没有找到有效匹配且路径非空，记录当前路径
        if not added and path:
            all_pathes.append(path)

    dfs(0, [])

    # 选择最长的路径作为最优匹配
    path = max(all_pathes, key=lambda x: len(x))

    # 重置所有 indices
    for it in toc_arr:
        it["indices"] = []
    # 应用最优路径
    for j, i in path:
        toc_arr[i]["indices"] = [j]

    # ========== 步骤4: 使用 LLM 填充未匹配的目录条目 ==========
    i = 0
    while i < len(toc_arr):
        it = toc_arr[i]
        # 如果已匹配，跳过
        if it["indices"]:
            i += 1
            continue

        # 确定搜索范围
        if i > 0 and toc_arr[i - 1]["indices"]:
            st_i = toc_arr[i - 1]["indices"][-1] + 1  # 从上一个匹配位置之后开始
        else:
            st_i = 0

        # 找到下一个已匹配的目录条目
        e = i + 1
        while e < len(toc_arr) and not toc_arr[e]["indices"]:
            e += 1
        if e >= len(toc_arr):
            e = len(sections)  # 到文档末尾
        else:
            e = toc_arr[e]["indices"][0]  # 到下一个匹配位置

        # 使用 LLM 在搜索范围内查找匹配
        for j in range(st_i, min(e + 1, len(sections))):
            ans = await gen_json(
                PROMPT_JINJA_ENV.from_string(TOC_INDEX).render(
                    structure=it["structure"],
                    title=it["title"],
                    text=sections[j]
                ), 
                "Only JSON please.", 
                chat_mdl
            )
            if ans["exist"] == "yes":
                it["indices"].append(j)
                break

        i += 1

    return toc_arr


async def check_if_toc_transformation_is_complete(content, toc, chat_mdl):
    """检查目录转换是否完成。

    使用 LLM 比较原始目录和清理后的目录，判断转换是否完整。

    Args:
        content (str): 原始目录内容
        toc (str): 清理后的目录内容
        chat_mdl: LLM 模型实例

    Returns:
        str: "yes" 或 "no"
    """
    prompt = """
    You are given a raw table of contents and a cleaned table of contents.
    Your job is to check if the cleaned table of contents is complete.

    Reply format:
    {{
        "thinking": <why do you think the cleaned table of contents is complete or not>,
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else.
    """

    # 组装完整提示词
    prompt = prompt + '\nRaw Table of contents:\n' + content + '\nCleaned Table of contents:\n' + toc
    response = await gen_json(prompt, "Only JSON please.", chat_mdl)
    return response['completed']


async def toc_transformer(toc_pages, chat_mdl):
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    The `structure` is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.
    The `title` is a short phrase or a several-words term.

    The response should be in the following JSON format:
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>
        },
        ...
    ],
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """

    toc_content = "\n".join(toc_pages)
    prompt = init_prompt + '\n Given table of contents\n:' + toc_content

    def clean_toc(arr):
        for a in arr:
            a["title"] = re.sub(r"[.·….]{2,}", "", a["title"])

    last_complete = await gen_json(prompt, "Only JSON please.", chat_mdl)
    if_complete = await check_if_toc_transformation_is_complete(toc_content,
                                                                json.dumps(last_complete, ensure_ascii=False, indent=2),
                                                                chat_mdl)
    clean_toc(last_complete)
    if if_complete == "yes":
        return last_complete

    while not (if_complete == "yes"):
        prompt = f"""
        Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.
        The response should be in the following JSON format:

        The raw table of contents json structure is:
        {toc_content}

        The incomplete transformed table of contents json structure is:
        {json.dumps(last_complete[-24:], ensure_ascii=False, indent=2)}

        Please continue the json structure, directly output the remaining part of the json structure."""
        new_complete = await gen_json(prompt, "Only JSON please.", chat_mdl)
        if not new_complete or str(last_complete).find(str(new_complete)) >= 0:
            break
        clean_toc(new_complete)
        last_complete.extend(new_complete)
        if_complete = await check_if_toc_transformation_is_complete(toc_content,
                                                                    json.dumps(last_complete, ensure_ascii=False,
                                                                               indent=2), chat_mdl)

    return last_complete


TOC_LEVELS = load_prompt("assign_toc_levels")


async def assign_toc_levels(toc_secs, chat_mdl, gen_conf={"temperature": 0.2}):
    if not toc_secs:
        return []
    return await gen_json(
        PROMPT_JINJA_ENV.from_string(TOC_LEVELS).render(),
        str(toc_secs),
        chat_mdl,
        gen_conf
    )


TOC_FROM_TEXT_SYSTEM = load_prompt("toc_from_text_system")
TOC_FROM_TEXT_USER = load_prompt("toc_from_text_user")


# 使用文本大模型从 chunk 列表中推断目录结构。
async def gen_toc_from_text(txt_info: dict, chat_mdl, callback=None):
    if callback:
        callback(msg="")
    try:
        ans = await gen_json(
            PROMPT_JINJA_ENV.from_string(TOC_FROM_TEXT_SYSTEM).render(),
            PROMPT_JINJA_ENV.from_string(TOC_FROM_TEXT_USER).render(
                text="\n".join([json.dumps(d, ensure_ascii=False) for d in txt_info["chunks"]])),
            chat_mdl,
            gen_conf={"temperature": 0.0, "top_p": 0.9}
        )
        txt_info["toc"] = ans if ans and not isinstance(ans, str) else []
    except Exception as e:
        logging.exception(e)


def split_chunks(chunks, max_length: int):
    """
    按 `max_length` 把 chunk 打包成若干批次，返回形如 `[{id: text}, ...]` 的结构。

    注意：
    - 这里不会把单个 chunk 再次切开；
    - 如果某个 chunk 自身就超过上限，也会原样放进单独批次。
    """

    result = []
    batch, batch_tokens = [], 0

    for idx, chunk in enumerate(chunks):
        t = num_tokens_from_string(chunk)
        if batch_tokens + t > max_length:
            result.append(batch)
            batch, batch_tokens = [], 0
        batch.append({idx: chunk})
        batch_tokens += t
    if batch:
        result.append(batch)
    return result


async def run_toc_from_text(chunks, chat_mdl, callback=None):
    input_budget = int(chat_mdl.max_length * INPUT_UTILIZATION) - num_tokens_from_string(
        TOC_FROM_TEXT_USER + TOC_FROM_TEXT_SYSTEM
    )

    input_budget = 1024 if input_budget > 1024 else input_budget
    chunk_sections = split_chunks(chunks, input_budget)
    titles = []

    chunks_res = []
    tasks = []
    for i, chunk in enumerate(chunk_sections):
        if not chunk:
            continue
        chunks_res.append({"chunks": chunk})
        tasks.append(asyncio.create_task(gen_toc_from_text(chunks_res[-1], chat_mdl, callback)))
    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        logging.error(f"Error generating TOC: {e}")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    for chunk in chunks_res:
        titles.extend(chunk.get("toc", []))

    # 过滤明显无效的标题项，例如 `title == -1` 的占位结果。
    prune = len(titles) > 512
    max_len = 12 if prune else 22
    filtered = []
    for x in titles:
        if not isinstance(x, dict) or not x.get("title") or x["title"] == "-1":
            continue
        if len(rag_tokenizer.tokenize(x["title"]).split(" ")) > max_len:
            continue
        if re.match(r"[0-9,.()/ -]+$", x["title"]):
            continue
        filtered.append(x)

    logging.info(f"\n\nFiltered TOC sections:\n{filtered}")
    if not filtered:
        return []

    # 先抽出标题文本，作为层级推断的输入。
    raw_structure = [x.get("title", "") for x in filtered]

    # 再让 LLM 预测每个标题所在的层级。
    toc_with_levels = await assign_toc_levels(raw_structure, chat_mdl, {"temperature": 0.0, "top_p": 0.9})
    if not toc_with_levels:
        return []

    # 最后按索引把“层级结构”和“标题内容”重新合并。
    prune = len(toc_with_levels) > 512
    max_lvl = "0"
    sorted_list = sorted([t.get("level", "0") for t in toc_with_levels if isinstance(t, dict)])
    if sorted_list:
        max_lvl = sorted_list[-1]
    merged = []
    for _, (toc_item, src_item) in enumerate(zip(toc_with_levels, filtered)):
        if prune and toc_item.get("level", "0") >= max_lvl:
            continue
        merged.append({
            "level": toc_item.get("level", "0"),
            "title": toc_item.get("title", ""),
            "chunk_id": src_item.get("chunk_id", ""),
        })

    return merged


TOC_RELEVANCE_SYSTEM = load_prompt("toc_relevance_system")
TOC_RELEVANCE_USER = load_prompt("toc_relevance_user")
async def relevant_chunks_with_toc(query: str, toc: list[dict], chat_mdl, topn: int = 6):
    import numpy as np
    try:
        ans = await gen_json(
            PROMPT_JINJA_ENV.from_string(TOC_RELEVANCE_SYSTEM).render(),
            PROMPT_JINJA_ENV.from_string(TOC_RELEVANCE_USER).render(query=query, toc_json="[\n%s\n]\n" % "\n".join(
                [json.dumps({"level": d["level"], "title": d["title"]}, ensure_ascii=False) for d in toc])),
            chat_mdl,
            gen_conf={"temperature": 0.0, "top_p": 0.9}
        )
        id2score = {}
        for ti, sc in zip(toc, ans):
            if not isinstance(sc, dict) or sc.get("score", -1) < 1:
                continue
            for id in ti.get("ids", []):
                if id not in id2score:
                    id2score[id] = []
                id2score[id].append(sc["score"] / 5.)
        for id in id2score.keys():
            id2score[id] = np.mean(id2score[id])
        return [(id, sc) for id, sc in list(id2score.items()) if sc >= 0.3][:topn]
    except Exception as e:
        logging.exception(e)
    return []


META_DATA = load_prompt("meta_data")
async def gen_metadata(chat_mdl, schema: dict, content: str):
    template = PROMPT_JINJA_ENV.from_string(META_DATA)
    for k, desc in schema["properties"].items():
        if "enum" in desc and not desc.get("enum"):
            del desc["enum"]
        if desc.get("enum"):
            desc["description"] += "\n** Extracted values must strictly match the given list specified by `enum`. **"
    system_prompt = template.render(content=content, schema=schema)
    user_prompt = "Output: "
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:])
    return re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)


SUFFICIENCY_CHECK = load_prompt("sufficiency_check")
async def sufficiency_check(chat_mdl, question: str, ret_content: str):
    try:
        return await gen_json(
            PROMPT_JINJA_ENV.from_string(SUFFICIENCY_CHECK).render(question=question, retrieved_docs=ret_content),
            "Output:\n",
            chat_mdl
        )
    except Exception as e:
        logging.exception(e)
    return {}


MULTI_QUERIES_GEN = load_prompt("multi_queries_gen")
async def multi_queries_gen(chat_mdl, question: str, query:str, missing_infos:list[str], ret_content: str):
    try:
        return await gen_json(
            PROMPT_JINJA_ENV.from_string(MULTI_QUERIES_GEN).render(
                original_question=question,
                original_query=query,
                missing_info="\n - ".join(missing_infos),
                retrieved_docs=ret_content
            ),
            "Output:\n",
            chat_mdl
        )
    except Exception as e:
        logging.exception(e)
    return {}
