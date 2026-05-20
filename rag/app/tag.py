"""标签能力占位模块。

原始工程里这里用于给问题或 chunk 打标签，作为检索排序的额外特征。
当前分支暂未保留完整实现，因此返回空标签列表。
"""


def label_question(question, knowledge_bases=None):
    """给问题打标签。

    当前分支返回空列表，表示不启用标签增强检索。
    """
    return []
