"""Prompt 子包导出入口。

这里把 `generator` 中的公开成员重新导出，便于外部直接从
`rag.prompts` 导入常用 prompt 辅助函数。
"""

from . import generator

__all__ = [name for name in dir(generator)
           if not name.startswith('_')]

globals().update({name: getattr(generator, name) for name in __all__})
