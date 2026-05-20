"""Tavily 检索客户端占位模块。

当前分支未接入真正的 Tavily Web Search，因此这里返回空结果，
只用于维持调用接口兼容。
"""

import logging


class Tavily:
    """Tavily 客户端占位实现。"""

    def __init__(self, api_key=None, *args, **kwargs):
        self.api_key = api_key

    def search(self, query, *args, **kwargs):
        """返回空搜索结果。"""
        return {"results": []}

    def get_search_context(self, query, *args, **kwargs):
        """返回空上下文。"""
        return ""
