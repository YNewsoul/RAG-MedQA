#
#  版权所有 2026 The InfiniFlow Authors。保留所有权利。
#
#  本文件遵循 Apache License 2.0 许可协议；
#  除非符合该许可协议，否则不得使用本文件。
#  许可协议全文见：
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  除非适用法律要求或书面同意，本软件按“原样”分发，
#  不提供任何明示或默示担保或条件。
#  具体权限和限制请参考上述许可协议。
#

"""聊天 SDK 路由层。

这里主要负责把前端请求转换成服务层调用，并把后端内部字段映射成前端接口字段。

本模块提供以下核心功能：
- 对话（Chat）的 CRUD 操作
- 会话（Session）的 CRUD 操作
- 问答接口（支持流式响应）

路由列表：
- GET    /chats                    - 获取对话列表
- POST   /chats                    - 创建对话
- GET    /chats/<chat_id>          - 获取单个对话
- PUT    /chats/<chat_id>          - 更新对话
- DELETE /chats/<chat_id>          - 删除对话
- GET    /chats/<chat_id>/sessions - 获取会话列表
- POST   /chats/<chat_id>/sessions - 创建会话
- GET    /chats/<chat_id>/sessions/<session_id> - 获取单个会话
- PUT    /chats/<chat_id>/sessions/<session_id> - 更新会话
- POST   /chats/ask                - 发送问答请求
"""

import asyncio
import json
import logging
from uuid import uuid4

from quart import request, Response

from api.apps import current_user, login_required
from api.db.db_models import Dialog, Conversation
from api.db.services.dialog_service import DialogService
from api.db.services.conversation_service import ConversationService, async_completion
from api.utils.api_utils import (
    get_data_error_result,
    get_json_result,
    get_request_json,
    server_error_response,
)
from common.constants import RetCode, StatusEnum, SYSTEM_TENANT_ID
from common.misc_utils import get_uuid



def _dialog_to_frontend(d):
    """把后端内部字段名转换成前端约定字段名。

    将数据库内部使用的字段名映射为前端接口约定的字段名，
    实现后端与前端的字段解耦。

    Args:
        d (dict): 后端对话数据字典

    Returns:
        dict: 转换后的前端格式数据字典
    """
    # 将内部字段 kb_ids（知识库ID列表）映射为前端字段 dataset_ids
    d["dataset_ids"] = d.pop("kb_ids", [])
    return d


@manager.route("/chats", methods=["GET"])  # noqa: F821
@login_required
def chats_list():
    """获取对话列表（分页）。

    查询系统中的对话列表，支持分页、排序和筛选。

    请求参数（Query）:
        page (int): 页码，默认值为 1
        page_size (int): 每页大小，默认值为 12
        orderby (str): 排序字段，默认值为 "update_time"
        desc (str): 是否降序，默认值为 "true"
        keywords (str): 搜索关键词，暂未使用
        id (str): 对话ID筛选
        name (str): 对话名称筛选

    Returns:
        dict: 包含对话列表和总数的 JSON 响应
            - chats: 对话列表（已转换为前端格式）
            - total: 对话总数
    """
    try:
        # 解析分页参数
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 12))
        # 解析排序参数
        orderby = request.args.get("orderby", "update_time")
        desc = request.args.get("desc", "true").lower() != "false"
        # 解析筛选参数
        keywords = request.args.get("keywords", "")
        chat_id = request.args.get("id")
        name = request.args.get("name")

        # 调用服务层获取对话列表
        dialogs, total = DialogService.get_list(
            tenant_id=SYSTEM_TENANT_ID,
            page_number=page,
            items_per_page=page_size,
            orderby=orderby,
            desc=desc,
            id=chat_id,
            name=name,
        )

        # 将后端格式转换为前端格式
        chats = [_dialog_to_frontend(d) for d in dialogs]
        return get_json_result(data={"chats": chats, "total": total})
    except Exception as e:
        return server_error_response(e)


@manager.route("/chats", methods=["POST"])  # noqa: F821
@login_required
async def chats_create():
    """创建新对话。

    创建一个新的对话配置，包括关联的知识库、LLM 模型和提示词配置。

    请求体（JSON）:
        name (str): 对话名称，默认值为 "New Chat"
        description (str): 对话描述，可选
        icon (str): 对话图标，可选
        dataset_ids (list): 关联的数据集/知识库ID列表
        llm_id (str): 关联的LLM模型ID
        prompt_config (dict): 提示词配置，可选

    Returns:
        dict: 创建成功的对话数据（已转换为前端格式）
    """
    try:
        # 获取请求体 JSON 数据
        req = await get_request_json()
        # 提取参数，设置默认值
        name = req.get("name", "New Chat")
        description = req.get("description", "")
        icon = req.get("icon", "")
        dataset_ids = req.get("dataset_ids", [])
        llm_id = req.get("llm_id", "")
        prompt_config = req.get("prompt_config", {})

        # 生成唯一对话ID
        chat_id = get_uuid()
        # 构建对话数据对象
        dialog = {
            "id": chat_id,
            "name": name,
            "description": description,
            "icon": icon,
            "kb_ids": dataset_ids,  # 内部字段名
            "llm_id": llm_id,
            "status": StatusEnum.VALID.value,  # 状态设为有效
        }
        # 如果提供了提示词配置，则添加到对话中
        if prompt_config:
            dialog["prompt_config"] = prompt_config

        # 保存到数据库
        DialogService.save(**dialog)
        # 验证保存结果并获取完整数据
        ok, chat = DialogService.get_by_id(chat_id)
        if not ok:
            return get_data_error_result(message="Failed to create chat")

        # 转换为前端格式并返回
        data = _dialog_to_frontend(chat.to_dict())
        return get_json_result(data=data)
    except Exception as e:
        return server_error_response(e)


@manager.route("/chats/<chat_id>", methods=["GET"])  # noqa: F821
@login_required
def chats_get(chat_id):
    """获取单个对话详情。

    根据对话ID查询对话的详细信息。

    路径参数:
        chat_id (str): 对话ID

    Returns:
        dict: 对话详情数据（已转换为前端格式）

    Error Responses:
        - 404: Chat not found
        - 403: No authorization（预留）
    """
    try:
        # 根据ID查询对话
        ok, chat = DialogService.get_by_id(chat_id)
        if not ok:
            return get_data_error_result(message="Chat not found")
        # 租户权限检查（预留，当前注释掉）
        if False:  # tenant_id removed
            return get_data_error_result(message="No authorization")
        # 转换为前端格式并返回
        data = _dialog_to_frontend(chat.to_dict())
        return get_json_result(data=data)
    except Exception as e:
        return server_error_response(e)


@manager.route("/chats/<chat_id>", methods=["PUT"])  # noqa: F821
@login_required
async def chats_update(chat_id):
    """更新对话配置。

    更新指定对话的配置信息，支持更新名称、描述、图标、关联数据集等。

    路径参数:
        chat_id (str): 对话ID

    请求体（JSON）:
        name (str): 对话名称
        description (str): 对话描述
        icon (str): 对话图标
        dataset_ids (list): 关联的数据集ID列表（前端字段）
        kb_ids (list): 关联的知识库ID列表（内部字段）
        llm_id (str): LLM模型ID
        prompt_config (dict): 提示词配置

    Returns:
        dict: 更新后的对话数据（已转换为前端格式）

    Error Responses:
        - 404: Chat not found
        - 403: No authorization（预留）
    """
    try:
        # 检查对话是否存在
        ok, chat = DialogService.get_by_id(chat_id)
        if not ok:
            return get_data_error_result(message="Chat not found")
        # 租户权限检查（预留）
        if False:  # tenant_id removed
            return get_data_error_result(message="No authorization")

        # 获取更新请求数据
        req = await get_request_json()
        # 字段映射：将前端的 dataset_ids 转换为内部的 kb_ids
        if "kb_ids" in req:
            req["kb_ids"] = req.pop("kb_ids")
        if "dataset_ids" in req:
            req["kb_ids"] = req.pop("dataset_ids")

        # 执行更新操作
        DialogService.update_by_id(chat_id, req)
        # 获取更新后的数据
        ok, updated = DialogService.get_by_id(chat_id)
        # 转换为前端格式并返回
        data = _dialog_to_frontend(updated.to_dict())
        return get_json_result(data=data)
    except Exception as e:
        return server_error_response(e)


@manager.route("/chats/<chat_id>", methods=["DELETE"])  # noqa: F821
@login_required
def chats_delete(chat_id):
    """删除对话。

    删除指定的对话及其关联的所有会话记录。

    路径参数:
        chat_id (str): 对话ID

    Returns:
        dict: {"data": true} 表示删除成功

    Error Responses:
        - 404: Chat not found
        - 403: No authorization（预留）
    """
    try:
        # 检查对话是否存在
        ok, chat = DialogService.get_by_id(chat_id)
        if not ok:
            return get_data_error_result(message="Chat not found")
        # 租户权限检查（预留）
        if False:  # tenant_id removed
            return get_data_error_result(message="No authorization")
        # 执行删除操作
        DialogService.delete_by_id(chat_id)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route("/chats/<chat_id>/sessions", methods=["GET"])  # noqa: F821
@login_required
def sessions_list(chat_id):
    """获取对话下的会话列表。

    查询指定对话下的所有会话记录，支持分页和筛选。

    路径参数:
        chat_id (str): 对话ID

    请求参数（Query）:
        page (int): 页码，默认值为 1
        page_size (int): 每页大小，默认值为 30
        orderby (str): 排序字段，默认值为 "update_time"
        desc (str): 是否降序，默认值为 "true"
        id (str): 会话ID筛选
        name (str): 会话名称筛选

    Returns:
        dict: 会话列表数据

    Error Responses:
        - 404: Chat not found
        - 403: No authorization（预留）
    """
    try:
        # 检查对话是否存在
        ok, chat = DialogService.get_by_id(chat_id)
        if not ok:
            return get_data_error_result(message="Chat not found")
        # 租户权限检查（预留）
        if False:  # tenant_id removed
            return get_data_error_result(message="No authorization")

        # 解析分页参数
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 30))
        # 解析排序参数
        orderby = request.args.get("orderby", "update_time")
        desc = request.args.get("desc", "true").lower() != "false"
        # 解析筛选参数
        session_id = request.args.get("id")
        name = request.args.get("name")

        # 调用服务层获取会话列表
        sessions = ConversationService.get_list(
            dialog_id=chat_id,
            page_number=page,
            items_per_page=page_size,
            orderby=orderby,
            desc=desc,
            id=session_id,
            name=name,
        )
        return get_json_result(data=sessions)
    except Exception as e:
        return server_error_response(e)


@manager.route("/chats/<chat_id>/sessions", methods=["POST"])  # noqa: F821
@login_required
async def sessions_create(chat_id):
    """创建新会话。

    在指定对话下创建一个新的会话（对话历史记录）。

    路径参数:
        chat_id (str): 对话ID

    请求体（JSON）:
        name (str): 会话名称，默认值为 "New session"

    Returns:
        dict: 创建的会话数据

    Error Responses:
        - 404: Chat not found
        - 403: No authorization（预留）
    """
    try:
        # 检查对话是否存在
        ok, chat = DialogService.get_by_id(chat_id)
        if not ok:
            return get_data_error_result(message="Chat not found")
        # 租户权限检查（预留）
        if False:  # tenant_id removed
            return get_data_error_result(message="No authorization")

        # 获取请求体数据
        req = await get_request_json()
        name = req.get("name", "New session")
        # 生成唯一会话ID
        session_id = get_uuid()

        # 构建会话数据对象
        conv = {
            "id": session_id,
            "dialog_id": chat_id,
            "name": name,
            "message": [],  # 初始化为空消息列表
            "user_id": current_user.id,  # 关联当前用户
        }
        # 保存到数据库
        ConversationService.save(**conv)
        return get_json_result(data=conv)
    except Exception as e:
        return server_error_response(e)


@manager.route("/chats/<chat_id>/sessions/<session_id>", methods=["GET"])  # noqa: F821
@login_required
def sessions_get(chat_id, session_id):
    """获取单个会话详情。

    根据对话ID和会话ID查询会话的详细信息，包括完整的消息历史。

    路径参数:
        chat_id (str): 对话ID
        session_id (str): 会话ID

    Returns:
        dict: 会话详情数据（包含消息历史）

    Error Responses:
        - 404: Session not found
    """
    try:
        # 根据ID和所属对话查询会话
        convs = ConversationService.query(id=session_id, dialog_id=chat_id)
        if not convs:
            return get_data_error_result(message="Session not found")
        # 返回第一个匹配的会话
        return get_json_result(data=convs[0].to_dict())
    except Exception as e:
        return server_error_response(e)


@manager.route("/chats/<chat_id>/sessions/<session_id>", methods=["PUT"])  # noqa: F821
@login_required
async def sessions_update(chat_id, session_id):
    """更新会话信息。

    更新指定会话的配置信息，如名称等。

    路径参数:
        chat_id (str): 对话ID
        session_id (str): 会话ID

    请求体（JSON）:
        name (str): 会话名称
        message (list): 消息列表（可用于修改消息历史）

    Returns:
        dict: 更新后的会话数据

    Error Responses:
        - 404: Session not found
    """
    try:
        # 检查会话是否存在
        convs = ConversationService.query(id=session_id, dialog_id=chat_id)
        if not convs:
            return get_data_error_result(message="Session not found")

        # 获取更新请求数据
        req = await get_request_json()
        # 执行更新操作
        ConversationService.update_by_id(session_id, req)
        # 获取更新后的数据
        convs = ConversationService.query(id=session_id, dialog_id=chat_id)
        return get_json_result(data=convs[0].to_dict())
    except Exception as e:
        return server_error_response(e)


@manager.route("/chats/ask", methods=["POST"])  # noqa: F821
@login_required
async def chats_ask():
    """发送问答请求（核心接口）。

    向指定对话发送问题，获取AI回答。支持流式响应（Server-Sent Events）。

    请求体（JSON）:
        question (str): 用户问题（必填）
        dialog_id (str): 对话ID（必填，与 chat_id 二选一）
        chat_id (str): 对话ID（别名，与 dialog_id 等效）
        conversation_id (str): 会话ID（可选，与 session_id 二选一）
        session_id (str): 会话ID（别名，与 conversation_id 等效）
        stream (bool): 是否流式响应，默认值为 true
        kb_ids (list): 临时指定的知识库ID列表（可选，覆盖对话默认配置）
        name (str): 新会话名称（当 session_id 为空时使用）

    Returns:
        Response: 流式响应（text/event-stream）或完整JSON响应

    Error Responses:
        - 400: question is required / chat_id is required
    """
    try:
        # 获取请求体数据
        req = await get_request_json()
        # 提取问题（必填）
        question = req.get("question", "")
        # 提取对话ID（支持两种字段名）
        chat_id = req.get("dialog_id") or req.get("chat_id", "")
        # 提取会话ID（支持两种字段名）
        session_id = req.get("conversation_id") or req.get("session_id")
        # 提取流式标志
        stream = req.get("stream", True)
        # 提取临时知识库列表
        kb_ids = req.get("kb_ids", [])
        # 新会话名称
        name = req.get("name", "New session")

        # 参数校验
        if not question:
            return get_data_error_result(message="question is required")
        if not chat_id:
            return get_data_error_result(message="chat_id is required")

        # 流式响应生成器
        async def generate():
            """异步生成流式响应内容。"""
            # 调用异步完成函数获取回答
            async for ans in async_completion(
                chat_id=chat_id,
                question=question,
                name=name,
                session_id=session_id,
                stream=stream,
                kb_ids=kb_ids,
            ):
                # 确保输出为字节流
                if isinstance(ans, str):
                    yield ans.encode("utf-8")
                else:
                    yield ans

        # 返回流式响应
        # text/event-stream: SSE（Server-Sent Events）格式
        # X-Accel-Buffering: no - 禁用Nginx缓冲，确保实时推送
        # Cache-Control: no-cache - 禁用缓存
        return Response(
            generate(),
            content_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
        )
    except Exception as e:
        return server_error_response(e)