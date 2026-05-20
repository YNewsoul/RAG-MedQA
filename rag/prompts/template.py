"""Prompt 模板加载器。

负责从 `rag/prompts/*.md` 中读取模板文本，并做简单缓存。
"""

import os

PROMPT_DIR = os.path.dirname(__file__)

_loaded_prompts = {}


def load_prompt(name: str) -> str:
    """按名称加载 Markdown Prompt 模板，并缓存结果。"""
    if name in _loaded_prompts:
        return _loaded_prompts[name]

    path = os.path.join(PROMPT_DIR, f"{name}.md")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file '{name}.md' not found in prompts/ directory.")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        _loaded_prompts[name] = content
        return content
