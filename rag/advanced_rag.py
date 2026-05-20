"""高级 RAG / 深度研究能力占位模块。

原始工程中这里可能承载多步检索、分阶段推理或 Deep Research 能力。
当前分支保留了最小接口，只用于不让上层调用报错。
"""

import logging


class DeepResearcher:
    """深度研究器占位实现。"""

    def __init__(self, chat_mdl, prompt_config, retrieval_fn, *args, **kwargs):
        self.chat_mdl = chat_mdl
        self.prompt_config = prompt_config
        self.retrieval_fn = retrieval_fn

    async def run(self, question, queue, *args, **kwargs):
        """向队列写入一个空结果，然后结束。"""
        await queue.put({"answer": "", "reference": {}})
        await queue.put(None)  # sentinel

    def __call__(self, *args, **kwargs):
        return self
