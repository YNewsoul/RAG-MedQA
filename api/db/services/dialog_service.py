"""对话服务层。

这个文件承接项目最核心的 RAG (Retrieval-Augmented Generation) 问答编排：

核心职责：
1. **模型绑定**：绑定聊天模型(LLM)、向量模型(Embedding)与重排模型(Rerank)
2. **问题处理**：执行问题改写、多轮对话整合、关键词补强
3. **检索编排**：支持向量检索、SQL检索、网页检索(Tavily)
4. **Prompt组装**：将检索到的知识片段组装成提示词
5. **引用增强**：在回答后自动补充引用标记和来源

关键函数说明：
- `async_chat()`: RAG 主链路入口，协调所有组件完成问答
- `async_chat_solo()`: 无知识库时的纯 LLM 对话
- `get_models()`: 获取对话所需的各类模型实例
- `use_sql()`: SQL 检索模式，支持结构化数据查询
- `structure_answer()`: 答案结构整理（在 conversation_service.py 中）

阅读建议：与 `conversation_service.py` 配套阅读，理解完整的问答流程。
"""

import asyncio
import binascii
import logging
import os
import re
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
from timeit import default_timer as timer

from peewee import fn
from api.db.services.file_service import FileService
from common.constants import LLMType, StatusEnum, SYSTEM_TENANT_ID
from api.db.db_models import DB, Dialog
from api.db.services.common_service import CommonService
from api.db.services.doc_metadata_service import DocMetadataService
from api.db.services.knowledgebase_service import KnowledgebaseService

from api.db.services.llm_service import LLMBundle
from common.metadata_utils import apply_meta_data_filter
from api.db.services.tenant_llm_service import TenantLLMService
from api.db.joint_services.tenant_model_service import get_model_config_by_id, get_model_config_by_type_and_name, get_tenant_default_model_by_type
from common.time_utils import current_timestamp, datetime_format
from common.text_utils import normalize_arabic_digits
from rag.advanced_rag import DeepResearcher
from rag.app.tag import label_question
from rag.nlp.search import index_name
from rag.prompts.generator import chunks_format, citation_prompt, cross_languages, full_question, kb_prompt, keyword_extraction, message_fit_in, \
    PROMPT_JINJA_ENV, ASK_SUMMARY
from common.token_utils import num_tokens_from_string
from rag.utils.tavily_conn import Tavily
from common.string_utils import remove_redundant_spaces
from common import settings


class DialogService(CommonService):
    """对话服务类，继承自 CommonService。

    提供对话（Dialog）的数据库操作，包括增删改查。
    对话是 RAG 系统的核心配置单元，包含知识库关联、模型配置、提示词配置等。
    """
    model = Dialog  # 指定数据库模型

    @classmethod
    def save(cls, **kwargs):
        """向数据库插入一条新的对话记录。
        Args:
            **kwargs: 对话字段键值对
        Returns:
            int: 插入记录的 ID
        """
        sample_obj = cls.model(**kwargs).save(force_insert=True)
        return sample_obj

    @classmethod
    def update_many_by_id(cls, data_list):
        """按 ID 批量更新记录，并刷新更新时间字段。

        Args:
            data_list (list[dict]): 要更新的记录列表，每条记录需包含 "id" 字段
        """
        with DB.atomic():
            for data in data_list:
                data["update_time"] = current_timestamp()
                data["update_date"] = datetime_format(datetime.now())
                cls.model.update(data).where(cls.model.id == data["id"]).execute()

    @classmethod
    @DB.connection_context()
    def get_list(cls, tenant_id, page_number, items_per_page, orderby, desc, id, name):
        """分页查询有效对话列表。

        Args:
            tenant_id (str): 租户ID
            page_number (int): 页码
            items_per_page (int): 每页大小
            orderby (str): 排序字段
            desc (bool): 是否降序
            id (str): 对话ID筛选（可选）
            name (str): 对话名称筛选（可选）

        Returns:
            tuple: (对话列表, 总数)
        """
        chats = cls.model.select()
        if id:
            chats = chats.where(cls.model.id == id)
        if name:
            chats = chats.where(cls.model.name == name)
        # 只查询有效状态的对话
        chats = chats.where(cls.model.status == StatusEnum.VALID.value)
        # 排序处理
        if desc:
            chats = chats.order_by(cls.model.getter_by(orderby).desc())
        else:
            chats = chats.order_by(cls.model.getter_by(orderby).asc())

        total = chats.count()
        chats = chats.paginate(page_number, items_per_page)

        return list(chats.dicts()), total

    @classmethod
    @DB.connection_context()
    def get_by_tenant_ids(
        cls,
        joined_tenant_ids,
        user_id,
        page_number,
        items_per_page,
        orderby,
        desc,
        keywords,
        id=None,
        name=None,
    ):
        fields = [
            cls.model.id,
            cls.model.name,
            cls.model.description,
            cls.model.language,
            cls.model.llm_id,
            cls.model.llm_setting,
            cls.model.prompt_type,
            cls.model.prompt_config,
            cls.model.similarity_threshold,
            cls.model.vector_similarity_weight,
            cls.model.top_n,
            cls.model.top_k,
            cls.model.do_refer,
            cls.model.rerank_id,
            cls.model.kb_ids,
            cls.model.icon,
            cls.model.status,
            cls.model.update_time,
            cls.model.create_time,
        ]
        dialogs = (
            cls.model.select(*fields)
            .where(cls.model.status == StatusEnum.VALID.value)
        )
        if id:
            dialogs = dialogs.where(cls.model.id == id)
        if name:
            dialogs = dialogs.where(cls.model.name == name)
        if keywords:
            dialogs = dialogs.where(fn.LOWER(cls.model.name).contains(keywords.lower()))
        if desc:
            dialogs = dialogs.order_by(cls.model.getter_by(orderby).desc())
        else:
            dialogs = dialogs.order_by(cls.model.getter_by(orderby).asc())

        count = dialogs.count()

        if page_number and items_per_page:
            dialogs = dialogs.paginate(page_number, items_per_page)

        return list(dialogs.dicts()), count

    @classmethod
    @DB.connection_context()
    def get_all_dialogs_by_tenant_id(cls, tenant_id):
        fields = [cls.model.id]
        dialogs = cls.model.select(*fields)
        dialogs.order_by(cls.model.create_time.asc())
        offset, limit = 0, 100
        res = []
        while True:
            d_batch = dialogs.offset(offset).limit(limit)
            _temp = list(d_batch.dicts())
            if not _temp:
                break
            res.extend(_temp)
            offset += limit
        return res

    @classmethod
    @DB.connection_context()
    def get_null_tenant_llm_id_row(cls):
        fields = [
            cls.model.id,
            cls.model.llm_id
        ]
        objs = cls.model.select(*fields).where(cls.model.tenant_llm_id.is_null())
        return list(objs)

    @classmethod
    @DB.connection_context()
    def get_null_tenant_rerank_id_row(cls):
        fields = [
            cls.model.id,
            cls.model.rerank_id
        ]
        objs = cls.model.select(*fields).where(cls.model.tenant_rerank_id.is_null())
        return list(objs)


async def async_chat_solo(dialog, messages, stream=True):
    """无知识库时的纯 LLM 对话。

    当对话没有绑定任何知识库且没有配置 Tavily 搜索时，
    直接调用 LLM 进行回答，不进行检索增强。

    Args:
        dialog (Dialog): 对话对象
        messages (list[dict]): 消息列表
        stream (bool): 是否流式响应

    Yields:
        dict: 包含 answer, reference, audio_binary 等字段的响应
    """
    # 获取 LLM 类型（chat 或 image2text）
    llm_type = TenantLLMService.llm_id2llm_type(dialog.llm_id)
    attachments = ""
    image_attachments = []
    image_files = []

    # 处理附件
    if "files" in messages[-1]:
        if llm_type == "chat":
            text_attachments, image_attachments = split_file_attachments(messages[-1]["files"])
        else:
            text_attachments, image_files = split_file_attachments(messages[-1]["files"], raw=True)
        attachments = "\n\n".join(text_attachments)

    # 获取聊天模型
    model_config = get_model_config_by_id(dialog.tenant_llm_id)
    chat_mdl = LLMBundle(model_config)
    factory = model_config.get("llm_factory", "") if model_config else ""

    # 初始化 TTS 模型（如果配置了）
    prompt_config = dialog.prompt_config
    tts_mdl = None
    if prompt_config.get("tts"):
        default_tts_model = get_tenant_default_model_by_type(LLMType.TTS)
        tts_mdl = LLMBundle(default_tts_model)

    # 构建消息列表（移除 system 消息和引用标记）
    msg = [{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])} 
           for m in messages if m["role"] != "system"]
    
    # 附加文件内容到最后一条消息
    if attachments and msg:
        msg[-1]["content"] += attachments

    # 处理多模态输入（图片）
    if llm_type == "chat" and image_attachments:
        convert_last_user_msg_to_multimodal(msg, image_attachments, factory)

    # 流式响应
    if stream:
        if llm_type == "chat":
            stream_iter = chat_mdl.async_chat_streamly_delta(
                prompt_config.get("system", ""), msg, dialog.llm_setting
            )
        else:
            stream_iter = chat_mdl.async_chat_streamly_delta(
                prompt_config.get("system", ""), msg, dialog.llm_setting, images=image_files
            )
        async for kind, value, state in _stream_with_think_delta(stream_iter):
            if kind == "marker":
                # 思考模式标记
                flags = {"start_to_think": True} if value == "<think>" else {"end_to_think": True}
                yield {
                    "answer": "", "reference": {}, "audio_binary": None, 
                    "prompt": "", "created_at": time.time(), "final": False, **flags
                }
                continue
            yield {
                "answer": value, "reference": {}, "audio_binary": tts(tts_mdl, value), 
                "prompt": "", "created_at": time.time(), "final": False
            }
    else:
        # 非流式响应
        if llm_type == "chat":
            answer = await chat_mdl.async_chat(
                prompt_config.get("system", ""), msg, dialog.llm_setting
            )
        else:
            answer = await chat_mdl.async_chat(
                prompt_config.get("system", ""), msg, dialog.llm_setting, images=image_files
            )
        user_content = msg[-1].get("content", "[content not available]")
        logging.debug("User: {}|Assistant: {}".format(user_content, answer))
        yield {
            "answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, answer), 
            "prompt": "", "created_at": time.time()
        }


def get_models(dialog):
    """获取对话所需的所有模型实例。

    根据对话配置，获取知识库、向量模型、重排模型、聊天模型和 TTS 模型。

    Args:
        dialog (Dialog): 对话对象

    Returns:
        tuple: (kbs, embd_mdl, rerank_mdl, chat_mdl, tts_mdl)
            - kbs: 知识库列表
            - embd_mdl: 向量模型（Embedding）
            - rerank_mdl: 重排模型
            - chat_mdl: 聊天模型（LLM）
            - tts_mdl: TTS 语音合成模型

    Raises:
        Exception: 多个知识库使用不同的嵌入模型
        LookupError: 嵌入模型未找到
    """
    embd_mdl, chat_mdl, rerank_mdl, tts_mdl = None, None, None, None

    # 获取关联的知识库
    kbs = KnowledgebaseService.get_by_ids(dialog.kb_ids)

    # 提取所有知识库使用的嵌入模型（必须一致）
    embedding_list = list(set([kb.embd_id for kb in kbs]))
    if len(embedding_list) > 1:
        raise Exception("**ERROR**: Knowledge bases use different embedding models.")

    # 初始化向量模型
    if embedding_list:
        embd_model_config = get_model_config_by_type_and_name(LLMType.EMBEDDING, embedding_list[0])
        embd_mdl = LLMBundle(embd_model_config)
        if not embd_mdl:
            raise LookupError("Embedding model(%s) not found" % embedding_list[0])

    # 初始化聊天模型（优先级：tenant_llm_id > llm_id > 默认模型）
    if dialog.tenant_llm_id:
        chat_model_config = get_model_config_by_id(dialog.tenant_llm_id)
    elif dialog.llm_id:
        chat_model_config = get_model_config_by_type_and_name(LLMType.CHAT, dialog.llm_id)
    else:
        chat_model_config = get_tenant_default_model_by_type(LLMType.CHAT)

    chat_mdl = LLMBundle(chat_model_config)

    # 初始化重排模型（可选）
    if dialog.rerank_id:
        rerank_model_config = get_model_config_by_type_and_name(LLMType.RERANK, dialog.rerank_id)
        rerank_mdl = LLMBundle(rerank_model_config)

    # 初始化 TTS 模型（可选）
    if dialog.prompt_config.get("tts"):
        default_tts_model_config = get_tenant_default_model_by_type(LLMType.TTS)
        tts_mdl = LLMBundle(default_tts_model_config)

    return kbs, embd_mdl, rerank_mdl, chat_mdl, tts_mdl


def split_file_attachments(files: list[dict] | None, raw: bool = False) -> tuple[list[str], list[str] | list[dict]]:
    """分离文件附件为文本和图片。

    将用户上传的文件附件分离为文本内容和图片内容两类。

    Args:
        files (list[dict]): 文件列表
        raw (bool): 是否返回原始文件对象（用于 image2text 模型）

    Returns:
        tuple: (text_attachments, image_attachments)
            - text_attachments: 文本内容列表
            - image_attachments: 图片内容列表（data URI 格式或原始对象）
    """
    if not files:
        return [], []

    text_attachments = []
    if raw:
        # image2text 模型需要原始文件对象
        file_contents, image_files = FileService.get_files(files, raw=True)
        for content in file_contents:
            if not isinstance(content, str):
                content = str(content)
            text_attachments.append(content)
        return text_attachments, image_files

    # chat 模型使用 data URI 格式
    image_attachments = []
    for content in FileService.get_files(files, raw=False):
        if not isinstance(content, str):
            content = str(content)
        # 判断是否为图片（data URI 格式）
        if content.strip().startswith("data:"):
            image_attachments.append(content.strip())
            continue
        text_attachments.append(content)
    return text_attachments, image_attachments


_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<b64>[A-Za-z0-9+/=\s]+)$")


def _parse_data_uri_or_b64(s: str, default_mime: str = "image/png") -> tuple[str, str]:
    s = (s or "").strip()
    match = _DATA_URI_RE.match(s)
    if match:
        mime = match.group("mime").strip()
        b64 = match.group("b64").strip()
        return mime, b64
    return default_mime, s


def _normalize_text_from_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for blk in content:
            if isinstance(blk, dict):
                if blk.get("type") in {"text", "input_text"}:
                    txt = blk.get("text")
                    if txt:
                        texts.append(str(txt))
                elif "text" in blk and isinstance(blk.get("text"), (str, int, float)):
                    texts.append(str(blk["text"]))
        return "\n".join(texts).strip()
    return str(content)


def convert_last_user_msg_to_multimodal(msg: list[dict], image_data_uris: list[str], factory: str) -> None:
    """将最后一条用户消息转换为多模态格式。

    根据不同的 LLM 工厂类型（Gemini、Anthropic、通用），
    将图片附件转换为对应的多模态消息格式。

    Args:
        msg (list[dict]): 消息列表
        image_data_uris (list[str]): 图片 data URI 列表
        factory (str): LLM 工厂名称（gemini、anthropic 等）
    """
    if not msg or not image_data_uris:
        return

    factory_norm = (factory or "").strip().lower()

    # 从后往前找到最后一条用户消息
    for idx in range(len(msg) - 1, -1, -1):
        if msg[idx].get("role") != "user":
            continue

        original_content = msg[idx].get("content", "")
        text = _normalize_text_from_content(original_content)

        # Gemini 格式
        if factory_norm == "gemini":
            parts = []
            if text:
                parts.append({"text": text})
            for image in image_data_uris:
                mime, b64 = _parse_data_uri_or_b64(str(image), default_mime="image/png")
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
            msg[idx]["content"] = parts
            return

        # Anthropic 格式
        if factory_norm == "anthropic":
            blocks = []
            if text:
                blocks.append({"type": "text", "text": text})
            for image in image_data_uris:
                mime, b64 = _parse_data_uri_or_b64(str(image), default_mime="image/png")
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                })
            msg[idx]["content"] = blocks
            return

        # 通用格式（OpenAI 风格）
        multimodal_content = []
        if isinstance(original_content, list):
            multimodal_content = deepcopy(original_content)
        else:
            text_content = "" if original_content is None else str(original_content)
            if text_content:
                multimodal_content.append({"type": "text", "text": text_content})

        for data_uri in image_data_uris:
            image_url = data_uri
            if not isinstance(image_url, str):
                image_url = str(image_url)
            if not image_url.startswith("data:"):
                image_url = f"data:image/png;base64,{image_url}"
            multimodal_content.append({"type": "image_url", "image_url": {"url": image_url}})

        msg[idx]["content"] = multimodal_content
        return


# 引用格式修复模式（处理模型输出的非标准引用格式）
BAD_CITATION_PATTERNS = [
    re.compile(r"\(\s*ID\s*[: ]*\s*(\d+)\s*\)"),  # (ID: 12)
    re.compile(r"\[\s*ID\s*[: ]*\s*(\d+)\s*\]"),  # [ID: 12]
    re.compile(r"【\s*ID\s*[: ]*\s*(\d+)\s*】"),  # 【ID: 12】
    re.compile(r"ref\s*(\d+)", flags=re.IGNORECASE),  # ref12、REF 12
]
# 标准引用标记模式
CITATION_MARKER_PATTERN = re.compile(r"\[(?:ID:)?([0-9\u0660-\u0669\u06F0-\u06F9]+)\]")


def repair_bad_citation_formats(answer: str, kbinfos: dict, idx: set):
    """修复模型输出中的非标准引用格式。

    将模型输出的各种非标准引用格式统一转换为标准格式 [ID:数字]，
    同时记录有效的引用索引。

    Args:
        answer (str): 模型回答文本
        kbinfos (dict): 知识库信息（包含 chunks）
        idx (set): 引用索引集合（用于记录有效的引用ID）

    Returns:
        tuple: (修复后的回答, 更新后的引用索引集合)
    """
    max_index = len(kbinfos["chunks"])
    normalized_answer = normalize_arabic_digits(answer) or ""

    # 安全添加索引（确保在有效范围内）
    def safe_add(i):
        if 0 <= i < max_index:
            idx.add(i)
            return True
        return False

    # 查找并替换引用格式
    def find_and_replace(pattern, group_index=1, repl=lambda digits: f"ID:{digits}"):
        nonlocal answer
        nonlocal normalized_answer

        matches = list(pattern.finditer(normalized_answer))
        if not matches:
            return

        parts = []
        last_idx = 0
        for match in matches:
            parts.append(answer[last_idx:match.start()])
            try:
                i = int(match.group(group_index))
            except Exception:
                parts.append(answer[match.start():match.end()])
                last_idx = match.end()
                continue

            if safe_add(i):
                digit_start, digit_end = match.span(group_index)
                digits_original = answer[digit_start:digit_end]
                parts.append(f"[{repl(digits_original)}]")
            else:
                parts.append(answer[match.start():match.end()])
            last_idx = match.end()

        parts.append(answer[last_idx:])
        answer = "".join(parts)
        normalized_answer = normalize_arabic_digits(answer) or ""

    # 应用所有修复模式
    for pattern in BAD_CITATION_PATTERNS:
        find_and_replace(pattern)

    return answer, idx


async def async_chat(dialog, messages, stream=True, **kwargs):
    """执行一次完整的 RAG (Retrieval-Augmented Generation) 对话主链路。

    RAG 主流程架构（9个阶段）：
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 阶段1: 输入校验 → 阶段2: 模型绑定 → 阶段3: 附件处理               │
    │         ↓                                                          │
    │ 阶段4: SQL检索(优先) → 阶段5: 问题增强 → 阶段6: 向量检索          │
    │         ↓                                                          │
    │ 阶段7: Prompt组装 → 阶段8: LLM推理 → 阶段9: 引用后处理           │
    └─────────────────────────────────────────────────────────────────────┘

    核心功能说明：
    - 支持多模态输入（文本、图片、文件附件）
    - 支持多种检索方式（向量检索、SQL检索、网页检索）
    - 支持深度研究模式（LLM驱动的多步推理）
    - 支持流式/非流式响应
    - 自动添加引用和性能统计

    Args:
        dialog (Dialog): 对话配置对象，包含以下关键属性：
            - kb_ids: 关联的知识库ID列表
            - llm_id/tenant_llm_id: LLM模型标识
            - rerank_id: 重排模型标识（可选）
            - prompt_config: 提示词配置（system提示、参数、引用设置等）
            - similarity_threshold: 向量相似度阈值
            - vector_similarity_weight: 向量相似度权重
            - top_n/top_k: 检索数量参数
            - meta_data_filter: 元数据过滤器

        messages (list[dict]): 对话消息列表，每条消息包含：
            - role: 角色（user/system/assistant）
            - content: 消息内容
            - files/doc_ids: 附件（可选）
            要求：最后一条消息必须是 user 角色

        stream (bool): 是否流式响应（SSE），默认为 True
            - True: 返回 Server-Sent Events 流式响应
            - False: 返回完整的单个响应

        **kwargs: 额外参数：
            - kb_ids: 临时指定的知识库ID列表（覆盖对话默认配置）
            - doc_ids: 指定的文档ID列表（用于限定检索范围）
            - files: 运行时附件文件
            - reasoning: 是否启用深度研究模式
            - quote: 是否添加引用
            - toolcall_session/tools: 工具调用配置

    Yields:
        dict: 响应字典，包含以下字段：
            - answer: 回答内容
            - reference: 引用信息（包含 chunks 和 doc_aggs）
            - audio_binary: TTS语音二进制数据（可选）
            - prompt: 完整提示词（含调试信息）
            - created_at: 创建时间戳
            - final: 是否为最终响应
            - start_to_think/end_to_think: 思考模式标记

    Raises:
        AssertionError: 最后一条消息不是用户消息
        KeyError: 缺少必填参数
        LookupError: 模型未找到
    """
    logging.debug("Begin async_chat - RAG main pipeline start")

    # ==================== 阶段1：输入校验 ====================
    # 断言：最后一条消息必须来自用户
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."

    # 快速路径：无知识库且无Tavily配置时，直接使用纯LLM对话
    if not dialog.kb_ids and not dialog.prompt_config.get("tavily_api_key"):
        logging.debug("No knowledge base and no Tavily configured, using LLM-only chat")
        async for ans in async_chat_solo(dialog, messages, stream):
            yield ans
        return

    # ==================== 阶段2：模型配置获取与绑定 ====================
    # 记录开始时间（用于性能统计）
    chat_start_ts = timer()

    # 获取LLM类型（chat 或 image2text）
    llm_type = TenantLLMService.llm_id2llm_type(dialog.llm_id)
    logging.debug(f"LLM type detected: {llm_type}")

    # 根据类型获取模型配置
    if llm_type == "image2text":
        llm_model_config = get_model_config_by_type_and_name(LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        llm_model_config = get_model_config_by_type_and_name(LLMType.CHAT, dialog.llm_id)

    # 提取模型工厂和最大token数
    factory = llm_model_config.get("llm_factory", "") if llm_model_config else ""
    max_tokens = llm_model_config.get("max_tokens", 8192)
    logging.debug(f"LLM factory: {factory}, max_tokens: {max_tokens}")

    check_llm_ts = timer()

    # 获取所有需要的模型实例
    # 返回：(知识库列表, 向量模型, 重排模型, 聊天模型, TTS模型)
    kbs, embd_mdl, rerank_mdl, chat_mdl, tts_mdl = get_models(dialog)
    logging.debug(f"Models loaded: embd={embd_mdl is not None}, rerank={rerank_mdl is not None}, chat={chat_mdl is not None}, tts={tts_mdl is not None}")

    # 如果配置了工具调用，绑定工具到聊天模型
    toolcall_session, tools = kwargs.get("toolcall_session"), kwargs.get("tools")
    if toolcall_session and tools:
        chat_mdl.bind_tools(toolcall_session, tools)
        logging.debug(f"Tools bound to chat model: {len(tools)} tools")

    bind_models_ts = timer()

    # ==================== 阶段3：问题提取与附件处理 ====================
    # 获取检索器实例
    retriever = settings.retriever

    # 提取最近3条用户消息行成 questions 列表（用于多轮对话上下文）
    questions = [m["content"] for m in messages if m["role"] == "user"][-3:]
    logging.debug(f"Extracted {len(questions)} user questions for retrieval")

    # 处理文档ID附件（从kwargs或消息中获取）
    attachments = None
    if "doc_ids" in kwargs:
        attachments = [doc_id for doc_id in kwargs["doc_ids"].split(",") if doc_id]

    # 处理文件附件，后续直接拼进系统 prompt，作为额外上下文
    attachments_ = ""  # 文本附件内容
    image_attachments = []  # chat模型用的图片（data URI格式）
    image_files = []  # image2text模型用的图片（原始格式）

    if "doc_ids" in messages[-1]:
        attachments = [doc_id for doc_id in messages[-1]["doc_ids"] if doc_id]

    if "files" in messages[-1]:
        if llm_type == "chat":
            # chat模型：分离文本和图片（data URI格式）
            text_attachments, image_attachments = split_file_attachments(messages[-1]["files"])
        else:
            # image2text模型：使用原始文件格式
            text_attachments, image_files = split_file_attachments(messages[-1]["files"], raw=True)
        attachments_ = "\n\n".join(text_attachments)
        logging.debug(f"Processed {len(text_attachments)} text attachments and {len(image_attachments + image_files)} image attachments")

    prompt_config = dialog.prompt_config

    # ==================== 阶段4：SQL检索（优先路径） ====================
    # 检查知识库是否有字段映射（结构化数据）
    field_map = KnowledgebaseService.get_field_map(dialog.kb_ids)
    logging.debug(f"Field map retrieved: {field_map}")

    # 如果存在字段映射，优先使用SQL检索（适用于结构化查询）
    if field_map:
        logging.debug(f"SQL retrieval available, query: {questions[-1]}")
        ans = await use_sql(
            question=questions[-1],
            field_map=field_map,
            tenant_id=SYSTEM_TENANT_ID,
            chat_mdl=chat_mdl,
            quota=prompt_config.get("quote", True),
            kb_ids=dialog.kb_ids
        )

        # 检查SQL结果：聚合查询（如COUNT、SUM）可能没有chunk但答案有效
        if ans and (ans.get("reference", {}).get("chunks") or ans.get("answer")):
            logging.debug("SQL retrieval successful, returning result")
            yield ans
            return
        else:
            logging.debug("SQL retrieval failed or returned no results, falling back to vector search")

    # ==================== 阶段5：检索前问题预处理：把原始问题加工成更适合检索的问题 ====================
    # 获取提示词参数列表
    param_keys = [p["key"] for p in prompt_config.get("parameters", [])]
    logging.debug(f"Prompt parameters: {param_keys}, attachments: {attachments}")

    # 5.1 参数校验：检查必填参数是否存在
    for p in prompt_config["parameters"]:
        if p["key"] == "knowledge":
            continue  # knowledge参数由系统自动填充
        # 必填参数缺失检查
        if p["key"] not in kwargs and not p["optional"]:
            raise KeyError("Missing required parameter: " + p["key"])
        # 可选参数未提供时，从系统提示词中移除占位符
        if p["key"] not in kwargs:
            prompt_config["system"] = prompt_config["system"].replace("{%s}" % p["key"], " ")

    # 5.2 多轮对话整合：将历史对话合并为完整问题
    if len(questions) > 1 and prompt_config.get("refine_multiturn"):
        logging.debug("Refining multi-turn conversation into single question")
        questions = [await full_question(dialog.llm_id, messages)]
    else:
        # 只保留最新一条问题
        questions = questions[-1:]

    # 5.3跨语言转换：将问题转换为目标语言（如中文转英文）
    if prompt_config.get("cross_languages"):
        target_lang = prompt_config["cross_languages"]
        logging.debug(f"Translating question to {target_lang}")
        questions = [await cross_languages(dialog.llm_id, questions[0], target_lang)]

    # 5.4 元数据过滤：根据元数据条件筛选文档
    if dialog.meta_data_filter:
        logging.debug("Applying metadata filter")
        metas = DocMetadataService.get_flatted_meta_by_kbs(dialog.kb_ids)
        attachments = await apply_meta_data_filter(
            dialog.meta_data_filter,
            metas,
            questions[-1],
            chat_mdl,
            attachments,
        )

    # 5.5 关键词补强：自动提取并追加关键词到问题中，提升检索准确性
    if prompt_config.get("keyword", False):
        logging.debug("Performing keyword extraction")
        keywords = await keyword_extraction(chat_mdl, questions[-1])
        questions[-1] += keywords
        logging.debug(f"Enhanced question with keywords: {questions[-1]}")

    refine_question_ts = timer()

    # ==================== 阶段6：检索执行 ====================
    thought = ""  # 思考内容存储（用于思考模式）
    # 检索结果容器
    kbinfos = {"total": 0, "chunks": [], "doc_aggs": []}  
    knowledges = []  # 后面会注入 prompt 的知识文本片段

    if "knowledge" in param_keys:
        logging.debug("Proceeding with retrieval")
        knowledges = []
        # ==================== 深度研究模式 ====================
        if prompt_config.get("reasoning", False) or kwargs.get("reasoning"):
            logging.debug("Enabling Deep Research mode for complex reasoning")
            # 深度研究模式：LLM驱动的多步推理，持续向前端输出思考过程
            reasoner = DeepResearcher(
                chat_mdl,
                prompt_config,
                partial(
                    retriever.retrieval,
                    embd_mdl=embd_mdl,
                    kb_ids=dialog.kb_ids,
                    page=1,
                    page_size=dialog.top_n,
                    similarity_threshold=0.2,
                    vector_similarity_weight=0.3,
                    doc_ids=attachments,
                ),
            )
            # 创建异步队列用于接收思考过程
            queue = asyncio.Queue()

            async def callback(msg:str):
                """回调函数：接收深度研究的中间结果"""
                nonlocal queue
                await queue.put(msg + "<br/>")

            await callback("<START_DEEP_RESEARCH>")
            task = asyncio.create_task(reasoner.research(kbinfos, questions[-1], questions[-1], callback=callback))
            while True:
                msg = await queue.get()
                if msg.find("<START_DEEP_RESEARCH>") == 0:
                    yield {"answer": "", "reference": {}, "audio_binary": None, "final": False, "start_to_think": True}
                elif msg.find("<END_DEEP_RESEARCH>") == 0:
                    yield {"answer": "", "reference": {}, "audio_binary": None, "final": False, "end_to_think": True}
                    break
                else:
                    yield {"answer": msg, "reference": {}, "audio_binary": None, "final": False}

            await task

        else:
            if embd_mdl:
                # 标准 RAG 路径：召回 chunk，必要时再做 TOC 增强和父子块合并。
                kbinfos = await retriever.retrieval(
                    " ".join(questions),
                    embd_mdl,
                    dialog.kb_ids,
                    1,
                    dialog.top_n,
                    dialog.similarity_threshold,
                    dialog.vector_similarity_weight,
                    doc_ids=attachments,
                    top=dialog.top_k,
                    aggs=True,
                    rerank_mdl=rerank_mdl,
                    rank_feature=label_question(" ".join(questions), kbs),
                )
                # 基于目录结构补召回
                if prompt_config.get("toc_enhance"):
                    cks = await retriever.retrieval_by_toc(" ".join(questions), kbinfos["chunks"], chat_mdl, dialog.top_n)
                    if cks:
                        kbinfos["chunks"] = cks
                # 命中子块时回收到父块，通常能给模型更完整的上下文。
                kbinfos["chunks"] = retriever.retrieval_by_children(kbinfos["chunks"])
            if prompt_config.get("tavily_api_key"):
                # 可选混入网页检索结果，作为外部知识补充。
                tav = Tavily(prompt_config["tavily_api_key"])
                tav_res = tav.retrieve_chunks(" ".join(questions))
                kbinfos["chunks"].extend(tav_res["chunks"])
                kbinfos["doc_aggs"].extend(tav_res["doc_aggs"])
    # 将检索到的知识片段格式化为prompt格式
    knowledges = kb_prompt(kbinfos, max_tokens)
    logging.debug("{}->{}".format(" ".join(questions), "\n->".join(knowledges)))

    retrieval_ts = timer()

    # 空结果处理：如果没有检索到知识且配置了空响应
    if not knowledges and prompt_config.get("empty_response"):
        empty_res = prompt_config["empty_response"]
        yield {
            "answer": empty_res, 
            "reference": kbinfos, 
            "prompt": "\n\n### Query:\n%s" % " ".join(questions),
            "audio_binary": tts(tts_mdl, empty_res), 
            "final": True
        }
        return

    # ==================== 阶段7：Prompt组装 ====================
    # 7.1 将知识片段添加到kwargs中
    kwargs["knowledge"] = "\n------\n" + "\n\n------\n\n".join(knowledges)

    # 7.2 获取LLM生成配置
    gen_conf = dialog.llm_setting

    # 7.3 构建系统提示词内容
    _system_content = prompt_config["system"].format(**kwargs) + attachments_ # 合并系统提示词和附件

    # 添加医疗免责声明（可通过环境变量关闭）
    if os.getenv("MEDICAL_DISCLAIMER_ENABLED", "true").lower() != "false":
        _system_content += (
            "\n\n---\n⚠️ 免责声明：本系统基于知识库提供医疗参考信息，不构成正式医疗建议，"
            "具体诊疗请以执业医师的专业判断为准。"
        )

    # 7.4 构建新的消息列表，第一条为系统提示词
    msg = [{"role": "system", "content": _system_content}]

    # 添加引用提示词（如果启用引用）
    prompt4citation = ""
    if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
        prompt4citation = citation_prompt()

    # 7.5 添加 添加用户消息（过滤掉系统消息和引用标记）
    msg.extend([
        {"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])}
        for m in messages if m["role"] != "system"
    ])

    # 7.6 确保消息token数在模型限制内
    used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.95))

    # 7.7 处理多模态消息（图片），把图片挂到最后一条 user message 上
    if llm_type == "chat" and image_attachments:
        convert_last_user_msg_to_multimodal(msg, image_attachments, factory)

    # 断言：确保至少有系统消息和一条用户消息
    assert len(msg) >= 2, f"message_fit_in has bug: {msg}"

    # 提取prompt内容（用于日志和调试）
    prompt = msg[0]["content"]

    # 计算可用的生成token数
    if "max_tokens" in gen_conf:
        gen_conf["max_tokens"] = min(gen_conf["max_tokens"], max_tokens - used_token_count)

    # ==================== 阶段8：答案后处理函数 ====================
    def decorate_answer(answer):
        """答案后处理函数：补引用、修复引用格式，并附加调试信息。
        把“模型原始输出”加工成“系统最终答案”

        主要职责：
        1. 分离思考块（<think>...</think>）- 提取LLM的内部推理过程
        2. 插入/修复引用标记 - 如果模型未显式给出引用，则通过相似度反推
        3. 过滤引用文档 - 根据引用的chunk筛选相关文档
        4. 添加性能统计信息 - 记录各阶段耗时和Token使用情况
        5. 处理API Key错误 - 提供友好的错误提示

        Args:
            answer (str): LLM 原始回答，可能包含 <think> 标签和引用标记

        Returns:
            dict: 包含以下字段的结果字典：
                - answer: 处理后的回答（含思考块）
                - reference: 引用信息（chunks和doc_aggs，移除了向量数据）
                - prompt: 完整提示词（含性能统计）
                - created_at: 创建时间戳
        """
        # 使用nonlocal声明以修改外层作用域的变量
        nonlocal embd_mdl, prompt_config, knowledges, kwargs, kbinfos, prompt, retrieval_ts, questions

        refs = []

        # ---------------------- 步骤1：分离思考块 ----------------------
        # 将回答拆分为思考块和实际答案
        ans = answer.split("</think>")
        think = ""
        if len(ans) == 2:
            think = ans[0] + "</think>"  # 保留思考块标签
            answer = ans[1]              # 提取实际答案

        # ---------------------- 步骤2：引用处理 ----------------------
        # 如果启用了引用，而且模型没自己正确打引用标记，就调用retriever插入引用
        if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
            idx = set([])  # 引用的chunk索引集合
            normalized_answer = normalize_arabic_digits(answer) or ""

            # 场景A：模型未显式给出引用标记，通过相似度反推
            if embd_mdl and not CITATION_MARKER_PATTERN.search(normalized_answer):
                answer, idx = retriever.insert_citations(
                    answer,
                    [ck["content_ltks"] for ck in kbinfos["chunks"]],
                    [ck["vector"] for ck in kbinfos["chunks"]],
                    embd_mdl,
                    tkweight=1 - dialog.vector_similarity_weight,  # 文本相似度权重
                    vtweight=dialog.vector_similarity_weight,       # 向量相似度权重
                    chunk_meta=kbinfos["chunks"],
                )
            # 场景B：模型已给出引用标记（如【1】【2】），直接提取
            else:
                for match in CITATION_MARKER_PATTERN.finditer(normalized_answer):
                    i = int(match.group(1))
                    if i < len(kbinfos["chunks"]):
                        idx.add(i)

            # 修复非标准引用格式（如编号错误、格式不一致等）
            answer, idx = repair_bad_citation_formats(answer, kbinfos, idx)

            # 根据引用的chunk筛选相关文档
            idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
            recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
            if not recall_docs:
                recall_docs = kbinfos["doc_aggs"]  # 兜底：使用所有文档
            kbinfos["doc_aggs"] = recall_docs

            # 构建引用结果（移除向量数据以减少传输大小）
            refs = deepcopy(kbinfos)
            for c in refs["chunks"]:
                if c.get("vector"):
                    del c["vector"]

        # ---------------------- 步骤3：错误处理 ----------------------
        # API Key错误处理：提供友好的错误提示
        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model providers -> API-Key'"

        # ---------------------- 步骤4：性能统计 ----------------------
        finish_chat_ts = timer()

        # 计算各阶段耗时（毫秒）
        total_time_cost = (finish_chat_ts - chat_start_ts) * 1000          # 总耗时
        check_llm_time_cost = (check_llm_ts - chat_start_ts) * 1000       # LLM配置检查耗时
        bind_embedding_time_cost = (bind_models_ts - check_llm_ts) * 1000  # 模型绑定耗时
        refine_question_time_cost = (refine_question_ts - bind_models_ts) * 1000  # 问题增强耗时
        retrieval_time_cost = (retrieval_ts - refine_question_ts) * 1000   # 检索耗时
        generate_result_time_cost = (finish_chat_ts - retrieval_ts) * 1000 # 答案生成耗时

        # 计算生成的Token数量
        tk_num = num_tokens_from_string(think + answer)

        # 将查询内容添加到prompt中（用于调试）
        prompt += "\n\n### Query:\n%s" % " ".join(questions)

        # 构建完整的性能统计信息
        prompt = (
            f"{prompt}\n\n"
            "## Time elapsed:\n"
            f"  - Total: {total_time_cost:.1f}ms\n"
            f"  - Check LLM: {check_llm_time_cost:.1f}ms\n"
            f"  - Bind models: {bind_embedding_time_cost:.1f}ms\n"
            f"  - Query refinement(LLM): {refine_question_time_cost:.1f}ms\n"
            f"  - Retrieval: {retrieval_time_cost:.1f}ms\n"
            f"  - Generate answer: {generate_result_time_cost:.1f}ms\n\n"
            "## Token usage:\n"
            f"  - Generated tokens(approximately): {tk_num}\n"
            f"  - Token speed: {int(tk_num / (generate_result_time_cost / 1000.0))}/s"
        )

        # 返回处理后的结果
        return {
            "answer": think + answer,  # 含思考块的完整回答
            "reference": refs,          # 引用信息
            "prompt": re.sub(r"\n", "  \n", prompt),  # 格式化的prompt（用于显示）
            "created_at": time.time()   # 创建时间戳
        }

    # ==================== 阶段9：LLM推理与响应生成 ====================

    # ==================== 流式响应模式（SSE）====================
    if stream:
        logging.debug("Using streaming response mode (SSE)")

        # 根据LLM类型调用不同的流式接口
        if llm_type == "chat":
            stream_iter = chat_mdl.async_chat_streamly_delta(
                prompt + prompt4citation,  # 完整prompt（含引用提示）
                msg[1:],                   # 用户消息（排除系统消息）
                gen_conf                   # 生成配置
            )
        else:
            # image2text模型需要传递图片
            stream_iter = chat_mdl.async_chat_streamly_delta(
                prompt + prompt4citation,
                msg[1:],
                gen_conf,
                images=image_files
            )

        last_state = None
        # 迭代处理流式输出（支持思考模式标记）
        async for kind, value, state in _stream_with_think_delta(stream_iter):
            last_state = state
            if kind == "marker":
                # 思考模式标记处理
                flags = {"start_to_think": True} if value == "<think>" else {"end_to_think": True}
                yield {
                    "answer": "", 
                    "reference": {}, 
                    "audio_binary": None, 
                    "final": False, 
                    **flags
                }
                continue
            # 输出文本内容（带TTS语音）
            yield {
                "answer": value, 
                "reference": {}, 
                "audio_binary": tts(tts_mdl, value), 
                "final": False
            }

        # 流式结束后，生成最终的引用增强答案
        full_answer = last_state.full_text if last_state else ""
        if full_answer:
            final = decorate_answer(thought + full_answer)
            final["final"] = True
            final["audio_binary"] = None  # 最终响应不包含TTS（流式阶段已逐段发送）
            yield final

    # ==================== 非流式响应模式 ====================
    else:
        logging.debug("Using non-streaming response mode")

        # 根据LLM类型调用不同的接口
        if llm_type == "chat":
            answer = await chat_mdl.async_chat(
                prompt + prompt4citation,
                msg[1:],
                gen_conf
            )
        else:
            # image2text模型需要传递图片
            answer = await chat_mdl.async_chat(
                prompt + prompt4citation,
                msg[1:],
                gen_conf,
                images=image_files
            )

        # 日志记录对话内容
        user_content = msg[-1].get("content", "[content not available]")
        logging.debug("User: {}|Assistant: {}".format(user_content, answer))
        # 答案后处理（添加引用、性能统计等）
        res = decorate_answer(answer)

        # 添加TTS语音
        res["audio_binary"] = tts(tts_mdl, answer)

        # 输出最终响应
        yield res

    # 函数结束
    return


async def use_sql(question, field_map, tenant_id, chat_mdl, quota=True, kb_ids=None):
    logging.debug(f"use_sql: Question: {question}")

    # Determine which document engine we're using
    if settings.DOC_ENGINE_INFINITY:
        doc_engine = "infinity"
    elif settings.DOC_ENGINE_OCEANBASE:
        doc_engine = "oceanbase"
    else:
        doc_engine = "es"

    # 构造完整表名。
    # Elasticsearch: 使用基础索引名，kb_id 通过 WHERE 条件过滤。
    # Infinity: 每个知识库独立成表，因此表名会额外拼上 kb_id。
    base_table = index_name()
    if doc_engine == "infinity" and kb_ids and len(kb_ids) == 1:
        # Infinity 模式下把 kb_id 追加到表名末尾。
        table_name = f"{base_table}_{kb_ids[0]}"
        logging.debug(f"use_sql: Using Infinity table name: {table_name}")
    else:
        # Elasticsearch/OpenSearch 使用基础索引名。
        table_name = base_table
        logging.debug(f"use_sql: Using ES/OS table name: {table_name}")

    expected_doc_name_column = "docnm" if doc_engine == "infinity" else "docnm_kwd"

    def has_source_columns(columns):
        normalized_names = {str(col.get("name", "")).lower() for col in columns}
        return "doc_id" in normalized_names and bool({"docnm_kwd", "docnm"} & normalized_names)

    def is_aggregate_sql(sql_text):
        return bool(re.search(r"(count|sum|avg|max|min|distinct)\s*\(", (sql_text or "").lower()))

    def normalize_sql(sql):
        logging.debug(f"use_sql: Raw SQL from LLM: {repr(sql[:500])}")
        # 去掉思考块内容（如 `</think>` 前后的中间结果）。
        sql = re.sub(r"</think>\n.*?\n\s*", "", sql, flags=re.DOTALL)
        sql = re.sub(r"思考\n.*?\n", "", sql, flags=re.DOTALL)
        # 去掉 markdown 代码块包裹。
        sql = re.sub(r"```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"```\s*$", "", sql, flags=re.IGNORECASE)
        # 去掉末尾分号，避免 ES SQL 解析器报错。
        return sql.rstrip().rstrip(';').strip()

    def add_kb_filter(sql):
        # 仅在 ES/OS 模式下追加 kb_id 过滤条件；Infinity 已经体现在表名里。
        if doc_engine == "infinity" or not kb_ids:
            return sql

        # 组装单知识库或多知识库过滤条件。
        if len(kb_ids) == 1:
            kb_filter = f"kb_id = '{kb_ids[0]}'"
        else:
            kb_filter = "(" + " OR ".join([f"kb_id = '{kb_id}'" for kb_id in kb_ids]) + ")"

        if "where " not in sql.lower():
            o = sql.lower().split("order by")
            if len(o) > 1:
                sql = o[0] + f" WHERE {kb_filter}  order by " + o[1]
            else:
                sql += f" WHERE {kb_filter}"
        elif "kb_id =" not in sql.lower() and "kb_id=" not in sql.lower():
            sql = re.sub(r"\bwhere\b ", f"where {kb_filter} and ", sql, flags=re.IGNORECASE)
        return sql

    def is_row_count_question(q: str) -> bool:
        q = (q or "").lower()
        if not re.search(r"\bhow many rows\b|\bnumber of rows\b|\brow count\b", q):
            return False
        return bool(re.search(r"\bdataset\b|\btable\b|\bspreadsheet\b|\bexcel\b", q))

    # 为不同文档引擎生成对应的 SQL prompt。
    if doc_engine == "infinity":
        # 为 Infinity 构造带 JSON 提取规则的 prompt。
        json_field_names = list(field_map.keys())
        row_count_override = (
            f"SELECT COUNT(*) AS rows FROM {table_name}"
            if is_row_count_question(question)
            else None
        )
        sys_prompt = """You are a Database Administrator. Write SQL for a table with JSON 'chunk_data' column.

JSON Extraction: json_extract_string(chunk_data, '$.FieldName')
Numeric Cast: CAST(json_extract_string(chunk_data, '$.FieldName') AS INTEGER/FLOAT)
NULL Check: json_extract_isnull(chunk_data, '$.FieldName') == false

RULES:
1. Use EXACT field names (case-sensitive) from the list below
2. For SELECT: include doc_id, docnm, and json_extract_string() for requested fields
3. For COUNT: use COUNT(*) or COUNT(DISTINCT json_extract_string(...))
4. Add AS alias for extracted field names
5. DO NOT select 'content' field
6. Only add NULL check (json_extract_isnull() == false) in WHERE clause when:
   - Question asks to "show me" or "display" specific columns
   - Question mentions "not null" or "excluding null"
   - Add NULL check for count specific column
   - DO NOT add NULL check for COUNT(*) queries (COUNT(*) counts all rows including nulls)
7. Output ONLY the SQL, no explanations"""
        user_prompt = """Table: {}
Fields (EXACT case): {}
{}
Question: {}
Write SQL using json_extract_string() with exact field names. Include doc_id, docnm for data queries. Only SQL.""".format(
            table_name,
            ", ".join(json_field_names),
            "\n".join([f"  - {field}" for field in json_field_names]),
            question
        )
    elif doc_engine == "oceanbase":
        # 为 OceanBase 构造带 JSON 提取规则的 prompt。
        json_field_names = list(field_map.keys())
        row_count_override = (
            f"SELECT COUNT(*) AS rows FROM {table_name}"
            if is_row_count_question(question)
            else None
        )
        sys_prompt = """You are a Database Administrator. Write SQL for a table with JSON 'chunk_data' column.

JSON Extraction: json_extract_string(chunk_data, '$.FieldName')
Numeric Cast: CAST(json_extract_string(chunk_data, '$.FieldName') AS INTEGER/FLOAT)
NULL Check: json_extract_isnull(chunk_data, '$.FieldName') == false

RULES:
1. Use EXACT field names (case-sensitive) from the list below
2. For SELECT: include doc_id, docnm_kwd, and json_extract_string() for requested fields
3. For COUNT: use COUNT(*) or COUNT(DISTINCT json_extract_string(...))
4. Add AS alias for extracted field names
5. DO NOT select 'content' field
6. Only add NULL check (json_extract_isnull() == false) in WHERE clause when:
   - Question asks to "show me" or "display" specific columns
   - Question mentions "not null" or "excluding null"
   - Add NULL check for count specific column
   - DO NOT add NULL check for COUNT(*) queries (COUNT(*) counts all rows including nulls)
7. Output ONLY the SQL, no explanations"""
        user_prompt = """Table: {}
Fields (EXACT case): {}
{}
Question: {}
Write SQL using json_extract_string() with exact field names. Include doc_id, docnm_kwd for data queries. Only SQL.""".format(
            table_name,
            ", ".join(json_field_names),
            "\n".join([f"  - {field}" for field in json_field_names]),
            question
        )
    else:
        # Build ES/OS prompts with direct field access
        row_count_override = None
        sys_prompt = """You are a Database Administrator. Write SQL queries.

RULES:
1. Use EXACT field names from the schema below (e.g., product_tks, not product)
2. Quote field names starting with digit: "123_field"
3. Add IS NOT NULL in WHERE clause when:
   - Question asks to "show me" or "display" specific columns
4. Include doc_id/docnm in non-aggregate statement
5. Output ONLY the SQL, no explanations"""
        user_prompt = """Table: {}
Available fields:
{}
Question: {}
Write SQL using exact field names above. Include doc_id, docnm_kwd for data queries. Only SQL.""".format(
            table_name,
            "\n".join([f"  - {k} ({v})" for k, v in field_map.items()]),
            question
        )

    tried_times = 0

    async def get_table(custom_user_prompt=None):
        nonlocal sys_prompt, user_prompt, question, tried_times, row_count_override
        if row_count_override and custom_user_prompt is None:
            sql = row_count_override
        else:
            prompt = custom_user_prompt if custom_user_prompt is not None else user_prompt
            sql = await chat_mdl.async_chat(sys_prompt, [{"role": "user", "content": prompt}], {"temperature": 0.06})
        sql = normalize_sql(sql)
        sql = add_kb_filter(sql)

        logging.debug(f"{question} get SQL(refined): {sql}")
        tried_times += 1
        logging.debug(f"use_sql: Executing SQL retrieval (attempt {tried_times})")
        tbl = settings.retriever.sql_retrieval(sql, format="json")
        if tbl is None:
            logging.debug("use_sql: SQL retrieval returned None")
            return None, sql
        logging.debug(f"use_sql: SQL retrieval completed, got {len(tbl.get('rows', []))} rows")
        return tbl, sql

    async def repair_table_for_missing_source_columns(previous_sql):
        if doc_engine in ("infinity", "oceanbase"):
            json_field_names = list(field_map.keys())
            repair_prompt = """Table name: {};
JSON fields available in 'chunk_data' column (use exact names):
{}

Question: {}
Previous SQL:
{}

The previous SQL result is missing required source columns for citations.
Rewrite SQL to keep the same query intent and include doc_id and {} in the SELECT list.
For extracted JSON fields, use json_extract_string(chunk_data, '$.field_name').
Return ONLY SQL.""".format(
                table_name,
                "\n".join([f"  - {field}" for field in json_field_names]),
                question,
                previous_sql,
                expected_doc_name_column
            )
        else:
            repair_prompt = """Table name: {}
Available fields:
{}

Question: {}
Previous SQL:
{}

The previous SQL result is missing required source columns for citations.
Rewrite SQL to keep the same query intent and include doc_id and docnm_kwd in the SELECT list.
Return ONLY SQL.""".format(
                table_name,
                "\n".join([f"  - {k} ({v})" for k, v in field_map.items()]),
                question,
                previous_sql
            )
        return await get_table(custom_user_prompt=repair_prompt)

    try:
        tbl, sql = await get_table()
        logging.debug(f"use_sql: Initial SQL execution SUCCESS. SQL: {sql}")
        logging.debug(f"use_sql: Retrieved {len(tbl.get('rows', []))} rows, columns: {[c['name'] for c in tbl.get('columns', [])]}")
    except Exception as e:
        logging.warning(f"use_sql: Initial SQL execution FAILED with error: {e}")
        # Build retry prompt with error information
        if doc_engine in ("infinity", "oceanbase"):
            # Build Infinity error retry prompt
            json_field_names = list(field_map.keys())
            user_prompt = """
Table name: {};
JSON fields available in 'chunk_data' column (use these exact names in json_extract_string):
{}

Question: {}
Please write the SQL using json_extract_string(chunk_data, '$.field_name') with the field names from the list above. Only SQL, no explanations.


The SQL error you provided last time is as follows:
{}

Please correct the error and write SQL again using json_extract_string(chunk_data, '$.field_name') syntax with the correct field names. Only SQL, no explanations.
""".format(table_name, "\n".join([f"  - {field}" for field in json_field_names]), question, e)
        else:
            # Build ES/OS error retry prompt
            user_prompt = """
        Table name: {};
        Table of database fields are as follows (use the field names directly in SQL):
        {}

        Question are as follows:
        {}
        Please write the SQL using the exact field names above, only SQL, without any other explanations or text.


        The SQL error you provided last time is as follows:
        {}

        Please correct the error and write SQL again using the exact field names above, only SQL, without any other explanations or text.
        """.format(table_name, "\n".join([f"{k} ({v})" for k, v in field_map.items()]), question, e)
        try:
            tbl, sql = await get_table()
            logging.debug(f"use_sql: Retry SQL execution SUCCESS. SQL: {sql}")
            logging.debug(f"use_sql: Retrieved {len(tbl.get('rows', []))} rows on retry")
        except Exception:
            logging.error("use_sql: Retry SQL execution also FAILED, returning None")
            return

    if len(tbl["rows"]) == 0:
        logging.warning(f"use_sql: No rows returned from SQL query, returning None. SQL: {sql}")
        return None

    if not is_aggregate_sql(sql) and not has_source_columns(tbl.get("columns", [])):
        logging.warning(f"use_sql: Non-aggregate SQL missing required source columns; retrying once. SQL: {sql}")
        try:
            repaired_tbl, repaired_sql = await repair_table_for_missing_source_columns(sql)
            if (
                repaired_tbl
                and len(repaired_tbl.get("rows", [])) > 0
                and has_source_columns(repaired_tbl.get("columns", []))
            ):
                tbl, sql = repaired_tbl, repaired_sql
                logging.info(f"use_sql: Source-column SQL repair succeeded. SQL: {sql}")
            else:
                logging.warning(f"use_sql: Source-column SQL repair did not provide required columns. Repaired SQL: {repaired_sql}")
        except Exception as e:
            logging.warning(f"use_sql: Source-column SQL repair failed, returning best-effort answer. Error: {e}")

    logging.debug(f"use_sql: Proceeding with {len(tbl['rows'])} rows to build answer")

    docid_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"].lower() == "doc_id"])
    doc_name_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"].lower() in ["docnm_kwd", "docnm"]])

    logging.debug(f"use_sql: All columns: {[(i, c['name']) for i, c in enumerate(tbl['columns'])]}")
    logging.debug(f"use_sql: docid_idx={docid_idx}, doc_name_idx={doc_name_idx}")

    column_idx = [ii for ii in range(len(tbl["columns"])) if ii not in (docid_idx | doc_name_idx)]

    logging.debug(f"use_sql: column_idx={column_idx}")
    logging.debug(f"use_sql: field_map={field_map}")

    # Helper function to map column names to display names
    def map_column_name(col_name):
        if col_name.lower() == "count(star)":
            return "COUNT(*)"

        # First, try to extract AS alias from any expression (aggregate functions, json_extract_string, etc.)
        # Pattern: anything AS alias_name
        as_match = re.search(r'\s+AS\s+([^\s,)]+)', col_name, re.IGNORECASE)
        if as_match:
            alias = as_match.group(1).strip('"\'')

            # Use the alias for display name lookup
            if alias in field_map:
                display = field_map[alias]
                return re.sub(r"(/.*|（[^（）]+）)", "", display)
            # If alias not in field_map, try to match case-insensitively
            for field_key, display_value in field_map.items():
                if field_key.lower() == alias.lower():
                    return re.sub(r"(/.*|（[^（）]+）)", "", display_value)
            # Return alias as-is if no mapping found
            return alias

        # Try direct mapping first (for simple column names)
        if col_name in field_map:
            display = field_map[col_name]
            # Clean up any suffix patterns
            return re.sub(r"(/.*|（[^（）]+）)", "", display)

        # Try case-insensitive match for simple column names
        col_lower = col_name.lower()
        for field_key, display_value in field_map.items():
            if field_key.lower() == col_lower:
                return re.sub(r"(/.*|（[^（）]+）)", "", display_value)

        # For aggregate expressions or complex expressions without AS alias,
        # try to replace field names with display names
        result = col_name
        for field_name, display_name in field_map.items():
            # Replace field_name with display_name in the expression
            result = result.replace(field_name, display_name)

        # Clean up any suffix patterns
        result = re.sub(r"(/.*|（[^（）]+）)", "", result)
        return result

    # compose Markdown table
    columns = (
            "|" + "|".join(
        [map_column_name(tbl["columns"][i]["name"]) for i in column_idx]) + (
                "|Source|" if docid_idx and doc_name_idx else "|")
    )

    line = "|" + "|".join(["------" for _ in range(len(column_idx))]) + ("|------|" if docid_idx and docid_idx else "")

    # Build rows ensuring column names match values - create a dict for each row
    # keyed by column name to handle any SQL column order
    rows = []
    for row_idx, r in enumerate(tbl["rows"]):
        row_dict = {tbl["columns"][i]["name"]: r[i] for i in range(len(tbl["columns"])) if i < len(r)}
        if row_idx == 0:
            logging.debug(f"use_sql: First row data: {row_dict}")
        row_values = []
        for col_idx in column_idx:
            col_name = tbl["columns"][col_idx]["name"]
            value = row_dict.get(col_name, " ")
            row_values.append(remove_redundant_spaces(str(value)).replace("None", " "))
        # Add Source column with citation marker if Source column exists
        if docid_idx and doc_name_idx:
            row_values.append(f" ##{row_idx}$$")
        row_str = "|" + "|".join(row_values) + "|"
        if re.sub(r"[ |]+", "", row_str):
            rows.append(row_str)
    if quota:
        rows = "\n".join(rows)
    else:
        rows = "\n".join(rows)
    rows = re.sub(r"T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+Z)?\|", "|", rows)

    if not docid_idx or not doc_name_idx:
        logging.warning(f"use_sql: SQL missing required doc_id or docnm_kwd field. docid_idx={docid_idx}, doc_name_idx={doc_name_idx}. SQL: {sql}")
        # For aggregate queries (COUNT, SUM, AVG, MAX, MIN, DISTINCT), fetch doc_id, docnm_kwd separately
        # to provide source chunks, but keep the original table format answer
        if is_aggregate_sql(sql):
            # Keep original table format as answer
            answer = "\n".join([columns, line, rows])

            # Now fetch doc_id, docnm_kwd to provide source chunks
            # Extract WHERE clause from the original SQL
            where_match = re.search(r"\bwhere\b(.+?)(?:\bgroup by\b|\border by\b|\blimit\b|$)", sql, re.IGNORECASE)
            if where_match:
                where_clause = where_match.group(1).strip()
                # Build a query to get doc_id and docnm_kwd with the same WHERE clause
                chunks_sql = f"select doc_id, docnm_kwd from {table_name} where {where_clause}"
                # Add LIMIT to avoid fetching too many chunks
                if "limit" not in chunks_sql.lower():
                    chunks_sql += " limit 20"
                logging.debug(f"use_sql: Fetching chunks with SQL: {chunks_sql}")
                try:
                    chunks_tbl = settings.retriever.sql_retrieval(chunks_sql, format="json")
                    if chunks_tbl.get("rows") and len(chunks_tbl["rows"]) > 0:
                        # Build chunks reference - use case-insensitive matching
                        chunks_did_idx = next((i for i, c in enumerate(chunks_tbl["columns"]) if c["name"].lower() == "doc_id"), None)
                        chunks_dn_idx = next((i for i, c in enumerate(chunks_tbl["columns"]) if c["name"].lower() in ["docnm_kwd", "docnm"]), None)
                        if chunks_did_idx is not None and chunks_dn_idx is not None:
                            chunks = [{"doc_id": r[chunks_did_idx], "docnm_kwd": r[chunks_dn_idx]} for r in chunks_tbl["rows"]]
                            # Build doc_aggs
                            doc_aggs = {}
                            for r in chunks_tbl["rows"]:
                                doc_id = r[chunks_did_idx]
                                doc_name = r[chunks_dn_idx]
                                if doc_id not in doc_aggs:
                                    doc_aggs[doc_id] = {"doc_name": doc_name, "count": 0}
                                doc_aggs[doc_id]["count"] += 1
                            doc_aggs_list = [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in doc_aggs.items()]
                            logging.debug(f"use_sql: Returning aggregate answer with {len(chunks)} chunks from {len(doc_aggs)} documents")
                            return {"answer": answer, "reference": {"chunks": chunks, "doc_aggs": doc_aggs_list}, "prompt": sys_prompt}
                except Exception as e:
                    logging.warning(f"use_sql: Failed to fetch chunks: {e}")
            # Fallback: return answer without chunks
            return {"answer": answer, "reference": {"chunks": [], "doc_aggs": []}, "prompt": sys_prompt}
        # Fallback to table format for other cases
        return {"answer": "\n".join([columns, line, rows]), "reference": {"chunks": [], "doc_aggs": []}, "prompt": sys_prompt}

    docid_idx = list(docid_idx)[0]
    doc_name_idx = list(doc_name_idx)[0]
    doc_aggs = {}
    for r in tbl["rows"]:
        if r[docid_idx] not in doc_aggs:
            doc_aggs[r[docid_idx]] = {"doc_name": r[doc_name_idx], "count": 0}
        doc_aggs[r[docid_idx]]["count"] += 1

    result = {
        "answer": "\n".join([columns, line, rows]),
        "reference": {
            "chunks": [{"doc_id": r[docid_idx], "docnm_kwd": r[doc_name_idx]} for r in tbl["rows"]],
            "doc_aggs": [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in doc_aggs.items()],
        },
        "prompt": sys_prompt,
    }
    logging.debug(f"use_sql: Returning answer with {len(result['reference']['chunks'])} chunks from {len(doc_aggs)} documents")
    return result

def clean_tts_text(text: str) -> str:
    if not text:
        return ""

    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U0001FAD0-\U0001FAFF]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    text = re.sub(r"\s+", " ", text).strip()

    MAX_LEN = 500
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN]

    return text

def tts(tts_mdl, text):
    if not tts_mdl or not text:
        return None
    text = clean_tts_text(text)
    if not text:
        return None
    bin = b""
    try:
        for chunk in tts_mdl.tts(text):
            bin += chunk
    except Exception as e:
        logging.error(f"TTS failed: {e}, text={text!r}")
        return None
    return binascii.hexlify(bin).decode("utf-8")


class _ThinkStreamState:
    def __init__(self) -> None:
        self.full_text = ""
        self.last_idx = 0
        self.endswith_think = False
        self.last_full = ""
        self.last_model_full = ""
        self.in_think = False
        self.buffer = ""


def _next_think_delta(state: _ThinkStreamState) -> str:
    full_text = state.full_text
    if full_text == state.last_full:
        return ""
    state.last_full = full_text
    delta_ans = full_text[state.last_idx:]

    if delta_ans.find("<think>") == 0:
        state.last_idx += len("<think>")
        return "<think>"
    if delta_ans.find("<think>") > 0:
        delta_text = full_text[state.last_idx:state.last_idx + delta_ans.find("<think>")]
        state.last_idx += delta_ans.find("<think>")
        return delta_text
    if delta_ans.endswith("</think>"):
        state.endswith_think = True
    elif state.endswith_think:
        state.endswith_think = False
        return "</think>"

    state.last_idx = len(full_text)
    if full_text.endswith("</think>"):
        state.last_idx -= len("</think>")
    return re.sub(r"(<think>|</think>)", "", delta_ans)


async def _stream_with_think_delta(stream_iter, min_tokens: int = 16):
    state = _ThinkStreamState()
    async for chunk in stream_iter:
        if not chunk:
            continue
        if chunk.startswith(state.last_model_full):
            new_part = chunk[len(state.last_model_full):]
            state.last_model_full = chunk
        else:
            new_part = chunk
            state.last_model_full += chunk
        if not new_part:
            continue
        state.full_text += new_part
        delta = _next_think_delta(state)
        if not delta:
            continue
        if delta in ("<think>", "</think>"):
            if delta == "<think>" and state.in_think:
                continue
            if delta == "</think>" and not state.in_think:
                continue
            if state.buffer:
                yield ("text", state.buffer, state)
                state.buffer = ""
            state.in_think = delta == "<think>"
            yield ("marker", delta, state)
            continue
        state.buffer += delta
        if num_tokens_from_string(state.buffer) < min_tokens:
            continue
        yield ("text", state.buffer, state)
        state.buffer = ""

    if state.buffer:
        yield ("text", state.buffer, state)
        state.buffer = ""
    if state.endswith_think:
        yield ("marker", "</think>", state)

async def async_ask(question, kb_ids, tenant_id, chat_llm_name=None, search_config={}):
    doc_ids = search_config.get("doc_ids", [])
    rerank_mdl = None
    kb_ids = search_config.get("kb_ids", kb_ids)
    chat_llm_name = search_config.get("chat_id", chat_llm_name)
    rerank_id = search_config.get("rerank_id", "")
    meta_data_filter = search_config.get("meta_data_filter")

    kbs = KnowledgebaseService.get_by_ids(kb_ids)
    embedding_list = list(set([kb.embd_id for kb in kbs]))

    retriever = settings.retriever
    embd_model_config = get_model_config_by_type_and_name(LLMType.EMBEDDING, embedding_list[0])
    embd_mdl = LLMBundle(embd_model_config)
    chat_model_config = get_model_config_by_type_and_name(LLMType.CHAT, chat_llm_name)
    chat_mdl = LLMBundle(chat_model_config)
    if rerank_id:
        rerank_model_config = get_model_config_by_type_and_name(LLMType.RERANK, rerank_id)
        rerank_mdl = LLMBundle(rerank_model_config)
    max_tokens = chat_mdl.max_length

    if meta_data_filter:
        metas = DocMetadataService.get_flatted_meta_by_kbs(kb_ids)
        doc_ids = await apply_meta_data_filter(meta_data_filter, metas, question, chat_mdl, doc_ids)

    kbinfos = await retriever.retrieval(
        question=question,
        embd_mdl=embd_mdl,
        kb_ids=kb_ids,
        page=1,
        page_size=12,
        similarity_threshold=search_config.get("similarity_threshold", 0.1),
        vector_similarity_weight=search_config.get("vector_similarity_weight", 0.3),
        top=search_config.get("top_k", 1024),
        doc_ids=doc_ids,
        aggs=True,
        rerank_mdl=rerank_mdl,
        rank_feature=label_question(question, kbs)
    )

    knowledges = kb_prompt(kbinfos, max_tokens)
    sys_prompt = PROMPT_JINJA_ENV.from_string(ASK_SUMMARY).render(knowledge="\n".join(knowledges))

    msg = [{"role": "user", "content": question}]

    def decorate_answer(answer):
        nonlocal knowledges, kbinfos, sys_prompt
        answer, idx = retriever.insert_citations(answer, [ck["content_ltks"] for ck in kbinfos["chunks"]], [ck["vector"] for ck in kbinfos["chunks"]],
                                                 embd_mdl, tkweight=0.7, vtweight=0.3)
        idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
        recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
        if not recall_docs:
            recall_docs = kbinfos["doc_aggs"]
        kbinfos["doc_aggs"] = recall_docs
        refs = deepcopy(kbinfos)
        for c in refs["chunks"]:
            if c.get("vector"):
                del c["vector"]

        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        refs["chunks"] = chunks_format(refs)
        return {"answer": answer, "reference": refs}

    stream_iter = chat_mdl.async_chat_streamly_delta(sys_prompt, msg, {"temperature": 0.1})
    last_state = None
    async for kind, value, state in _stream_with_think_delta(stream_iter):
        last_state = state
        if kind == "marker":
            flags = {"start_to_think": True} if value == "<think>" else {"end_to_think": True}
            yield {"answer": "", "reference": {}, "final": False, **flags}
            continue
        yield {"answer": value, "reference": {}, "final": False}
    full_answer = last_state.full_text if last_state else ""
    final = decorate_answer(full_answer)
    final["final"] = True
    final["answer"] = ""
    yield final