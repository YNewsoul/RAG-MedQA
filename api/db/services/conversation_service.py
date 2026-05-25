"""会话服务层。

负责维护会话消息、引用结构以及 SSE (Server-Sent Events) 流式返回格式。

核心功能：
1. 会话的 CRUD 操作（基于 CommonService 继承）
2. 会话列表查询（支持分页、排序、筛选）
3. 答案结构整理（structure_answer）
4. 异步问答完成（async_completion）- 主链路入口
5. iframe 模式异步问答（async_iframe_completion）

阅读 RAG 主链路时，这个文件适合和 `dialog_service.py` 配套来看。
"""

import time
import logging
from uuid import uuid4
from common.constants import StatusEnum
from api.db.db_models import Conversation, DB
from api.db.services.api_service import API4ConversationService
from api.db.services.common_service import CommonService
from api.db.services.dialog_service import DialogService, async_chat
from common.misc_utils import get_uuid
import json

from rag.prompts.generator import chunks_format


class ConversationService(CommonService):
    """会话服务类，继承自 CommonService。

    提供会话（Conversation）的数据库操作，包括增删改查。
    """
    model = Conversation  # 指定数据库模型

    @classmethod
    @DB.connection_context()
    def get_list(cls, dialog_id, page_number, items_per_page, orderby, desc, id, name, user_id=None):
        """分页查询指定对话下的会话列表。

        Args:
            dialog_id (str): 对话ID（必填）
            page_number (int): 页码
            items_per_page (int): 每页大小（<=0 时不分页）
            orderby (str): 排序字段
            desc (bool): 是否降序
            id (str): 会话ID筛选（可选）
            name (str): 会话名称筛选（可选）
            user_id (str): 用户ID筛选（可选）

        Returns:
            list[dict]: 会话列表（字典格式）
        """
        # 基础查询：按对话ID筛选
        sessions = cls.model.select().where(cls.model.dialog_id == dialog_id)
        
        # 条件筛选
        if id:
            sessions = sessions.where(cls.model.id == id)
        if name:
            sessions = sessions.where(cls.model.name == name)
        if user_id:
            sessions = sessions.where(cls.model.user_id == user_id)
        
        # 排序处理
        if desc:
            sessions = sessions.order_by(cls.model.getter_by(orderby).desc())
        else:
            sessions = sessions.order_by(cls.model.getter_by(orderby).asc())

        # 分页处理（items_per_page <= 0 时返回全部）
        if items_per_page > 0:
            sessions = sessions.paginate(page_number, items_per_page)

        return list(sessions.dicts())

    @classmethod
    @DB.connection_context()
    def get_all_conversation_by_dialog_ids(cls, dialog_ids):
        """批量查询多个对话下的所有会话。

        使用分批查询避免一次性加载过多数据导致内存问题。

        Args:
            dialog_ids (list[str]): 对话ID列表

        Returns:
            list[dict]: 所有匹配的会话列表
        """
        # 查询指定对话ID列表下的所有会话，按创建时间升序
        sessions = cls.model.select().where(cls.model.dialog_id.in_(dialog_ids))
        sessions.order_by(cls.model.create_time.asc())
        
        # 分批查询（每批100条）
        offset, limit = 0, 100
        res = []
        while True:
            s_batch = sessions.offset(offset).limit(limit)
            _temp = list(s_batch.dicts())
            if not _temp:
                break
            res.extend(_temp)
            offset += limit
        return res


def structure_answer(conv, ans, message_id, session_id):
    """把模型输出整理成前端统一消费的答案结构。

    负责将 RAG 引擎返回的原始答案转换为前端期望的格式，
    同时更新会话对象中的消息历史和引用信息。

    Args:
        conv (Conversation): 会话对象（可为 None）
        ans (dict): RAG 引擎返回的原始答案
        message_id (str): 消息ID
        session_id (str): 会话ID

    Returns:
        dict: 格式化后的答案结构

    答案结构字段说明:
        - id: 消息ID
        - session_id: 会话ID
        - answer: 回答内容
        - reference: 引用信息（包含 chunks 和 doc_aggs）
        - audio_binary: 音频二进制数据（可选）
    """
    # 处理引用信息
    reference = ans["reference"]
    if not isinstance(reference, dict):
        reference = {}
        ans["reference"] = {}
    
    # 判断是否为最终答案
    is_final = ans.get("final", True)

    # 格式化引用片段
    chunk_list = chunks_format(reference)
    reference["chunks"] = chunk_list
    
    # 设置消息ID和会话ID
    ans["id"] = message_id
    ans["session_id"] = session_id

    # 如果没有会话对象，直接返回格式化后的答案
    if not conv:
        return ans

    # 初始化消息列表
    if not conv.message:
        conv.message = []
    
    # 处理思考模式标记
    content = ans["answer"]
    if ans.get("start_to_think"):
        content = "<think>"  # 开始思考标记
    elif ans.get("end_to_think"):
        content = "</think>"  # 结束思考标记

    # 更新会话消息历史
    if not conv.message or conv.message[-1].get("role", "") != "assistant":
        # 新建助手消息
        conv.message.append({
            "role": "assistant", 
            "content": content, 
            "created_at": time.time(), 
            "id": message_id
        })
    else:
        # 更新现有助手消息
        if is_final:
            # 最终答案：替换或更新
            if ans.get("answer"):
                conv.message[-1] = {
                    "role": "assistant", 
                    "content": ans["answer"], 
                    "created_at": time.time(), 
                    "id": message_id
                }
            else:
                conv.message[-1]["created_at"] = time.time()
                conv.message[-1]["id"] = message_id
        else:
            # 流式答案：追加内容
            conv.message[-1]["content"] = (conv.message[-1].get("content") or "") + content
            conv.message[-1]["created_at"] = time.time()
            conv.message[-1]["id"] = message_id
    
    # 更新引用信息
    if conv.reference:
        should_update_reference = is_final or bool(reference.get("chunks")) or bool(reference.get("doc_aggs"))
        if should_update_reference:
            conv.reference[-1] = reference
    
    return ans


async def async_completion(chat_id, question, name="New session", session_id=None, stream=True, **kwargs):
    """创建/续用会话，并把请求转发给 `async_chat`。

    这是问答主链路的核心入口函数，负责：
    1. 创建新会话或复用现有会话
    2. 构建消息上下文
    3. 调用 RAG 引擎进行问答
    4. 处理流式/非流式响应

    Args:
        chat_id (str): 对话ID
        question (str): 用户问题
        name (str): 会话名称（新建会话时使用），默认 "New session"
        session_id (str): 会话ID（可选，为空时创建新会话）
        stream (bool): 是否流式响应，默认 True
        **kwargs: 额外参数（如 kb_ids, user_id, files 等）

    Yields:
        str: SSE 格式的响应数据（流式模式）
        dict: 完整答案（非流式模式）

    Raises:
        AssertionError: name 为空或 chat_id 无效
        LookupError: session_id 指定的会话不存在
    """
    # 参数校验
    assert name, "`name` can not be empty."
    dia = DialogService.query(id=chat_id, status=StatusEnum.VALID.value)
    assert dia, "You do not own the chat."

    # ==================== 新建会话逻辑 ====================
    if not session_id:
        # 首次进入会话时，创建新会话并返回开场白即结束
        session_id = get_uuid()
        conv = {
            "id": session_id,
            "dialog_id": chat_id,
            "name": name,
            "message": [{"role": "assistant", "content": dia[0].prompt_config.get("prologue", ""), "created_at": time.time()}],
            "user_id": kwargs.get("user_id", "")
        }
        ConversationService.save(**conv)
        
        # 流式模式：返回开场白后结束
        if stream:
            yield "data:" + json.dumps({
                "code": 0, 
                "message": "",
                "data": {
                    "answer": conv["message"][0]["content"],
                    "reference": {},
                    "audio_binary": None,
                    "id": None,
                    "session_id": session_id
                }
            }, ensure_ascii=False) + "\n\n"
            # 发送结束标记
            yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"
            return
        else:
            # 非流式模式：直接返回开场白
            answer = {
                "answer": conv["message"][0]["content"],
                "reference": {},
                "audio_binary": None,
                "id": None,
                "session_id": session_id
            }
            yield answer
            return

    # ==================== 续用现有会话 ====================
    # 查询会话
    conv_filters = {"id": session_id, "dialog_id": chat_id}
    if kwargs.get("user_id"):
        conv_filters["user_id"] = kwargs["user_id"]
    conv = ConversationService.query(**conv_filters)
    if not conv:
        raise LookupError("Session does not exist")
    conv = conv[0]

    # 构建消息上下文
    msg = []
    question = {
        "content": question,
        "role": "user",
        "id": str(uuid4())
    }

    # 透传运行期附件，便于下游对话链路解析文件内容
    if isinstance(kwargs.get("files"), list) and kwargs["files"]:
        question["files"] = kwargs["files"]

    # 将问题添加到会话消息
    conv.message.append(question)
    
    # 过滤消息，过滤掉 system 消息，还会跳过第一条 assistant 开场白
    for m in conv.message:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant" and not msg:
            continue
        msg.append(m)
    message_id = msg[-1].get("id") # 获取最后一个消息的message_id，用于后续引用

    # 获取对话配置
    e, dia = DialogService.get_by_id(conv.dialog_id)

    # 合并知识库ID（对话默认 + 请求传入）
    kb_ids = kwargs.get("kb_ids", [])
    dia.kb_ids = list(set(dia.kb_ids + kb_ids))

    # 初始化引用列表
    if not conv.reference:
        conv.reference = []
    # 塞一个空的 assistant 占位消息，用于后续引用
    conv.message.append({"role": "assistant", "content": "", "id": str(uuid4())})
    # 塞一个空引用槽位，后面的流式 token 会不断往这个占位消息里追加
    conv.reference.append({"chunks": [], "doc_aggs": []})

    # ==================== 流式响应处理 ====================
    if stream:
        try:
            # 调用 RAG 引擎进行异步问答
            async for ans in async_chat(dia, msg, True, **kwargs):
                # 格式化答案
                ans = structure_answer(conv, ans, message_id, session_id)
                # 发送 SSE 格式数据
                yield "data:" + json.dumps({"code": 0, "data": ans}, ensure_ascii=False) + "\n\n"
            # 保存会话更新
            ConversationService.update_by_id(conv.id, conv.to_dict())
        except Exception as e:
            # 异常处理：发送错误消息
            logging.error("async_completion error: %s", str(e))
            yield "data:" + json.dumps({
                "code": 500, 
                "message": str(e),
                "data": {"answer": "**ERROR**: " + str(e), "reference": []}
            }, ensure_ascii=False) + "\n\n"
        # 发送结束标记
        yield "data:" + json.dumps({"code": 0, "data": True}, ensure_ascii=False) + "\n\n"

    # ==================== 非流式响应处理 ====================
    else:
        answer = None
        async for ans in async_chat(dia, msg, False, **kwargs):
            answer = structure_answer(conv, ans, message_id, session_id)
            ConversationService.update_by_id(conv.id, conv.to_dict())
            break
        yield answer

async def async_iframe_completion(dialog_id, question, session_id=None, stream=True, **kwargs):
    """iframe 嵌入模式的异步问答完成函数。

    与 `async_completion` 类似，但使用 `API4ConversationService` 而非 `ConversationService`，
    适用于外部系统通过 iframe 嵌入使用的场景。

    Args:
        dialog_id (str): 对话ID
        question (str): 用户问题
        session_id (str): 会话ID（可选）
        stream (bool): 是否流式响应，默认 True
        **kwargs: 额外参数

    Yields:
        str: SSE 格式的响应数据（流式模式）
        dict: 完整答案（非流式模式）

    Raises:
        AssertionError: dialog_id 无效或 session_id 不存在
    """
    # 验证对话存在
    e, dia = DialogService.get_by_id(dialog_id)
    assert e, "Dialog not found"

    # ==================== 新建会话 ====================
    if not session_id:
        session_id = get_uuid()
        conv = {
            "id": session_id,
            "dialog_id": dialog_id,
            "user_id": kwargs.get("user_id", ""),
            "message": [{"role": "assistant", "content": dia.prompt_config["prologue"], "created_at": time.time()}]
        }
        # 使用 API4ConversationService 保存
        API4ConversationService.save(**conv)
        
        # 返回开场白
        yield "data:" + json.dumps({
            "code": 0, 
            "message": "",
            "data": {
                "answer": conv["message"][0]["content"],
                "reference": {},
                "audio_binary": None,
                "id": None,
                "session_id": session_id
            }
        }, ensure_ascii=False) + "\n\n"
        yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"
        return
    
    # ==================== 续用现有会话 ====================
    else:
        e, conv = API4ConversationService.get_by_id(session_id)
        assert e, "Session not found!"

    # 初始化消息列表
    if not conv.message:
        conv.message = []
    messages = conv.message

    # 构建用户问题
    question = {
        "role": "user",
        "content": question,
        "id": str(uuid4())
    }
    messages.append(question)

    # 过滤消息上下文
    msg = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant" and not msg:
            continue
        msg.append(m)
    if not msg[-1].get("id"):
        msg[-1]["id"] = get_uuid()
    message_id = msg[-1]["id"]

    # 初始化引用列表
    if not conv.reference:
        conv.reference = []
    conv.reference.append({"chunks": [], "doc_aggs": []})

    # ==================== 流式响应 ====================
    if stream:
        try:
            async for ans in async_chat(dia, msg, True, **kwargs):
                ans = structure_answer(conv, ans, message_id, session_id)
                yield "data:" + json.dumps({"code": 0, "message": "", "data": ans},
                                           ensure_ascii=False) + "\n\n"
            API4ConversationService.append_message(conv.id, conv.to_dict())
        except Exception as e:
            yield "data:" + json.dumps({
                "code": 500, 
                "message": str(e),
                "data": {"answer": "**ERROR**: " + str(e), "reference": []}
            }, ensure_ascii=False) + "\n\n"
        yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

    # ==================== 非流式响应 ====================
    else:
        answer = None
        async for ans in async_chat(dia, msg, False, **kwargs):
            answer = structure_answer(conv, ans, message_id, session_id)
            API4ConversationService.append_message(conv.id, conv.to_dict())
            break
        yield answer
