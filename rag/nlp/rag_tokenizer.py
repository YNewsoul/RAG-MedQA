"""Tokenizer 兼容层。

底层主要复用 `infinity.rag_tokenizer`，但会根据当前文档引擎类型决定
是否真的执行分词，从而兼容 Infinity 的原生检索模式。

核心设计目的：
1. 当使用 Infinity 文档引擎时，跳过分词（Infinity 内置分词）
2. 当使用其他引擎时，执行标准分词处理
3. 通过延迟导入避免循环依赖
"""

import infinity.rag_tokenizer


class RagTokenizer(infinity.rag_tokenizer.RagTokenizer):
    """RAG 分词器类，支持多文档引擎兼容。"""

    def tokenize(self, line: str) -> str:
        """分词处理（根据文档引擎类型决定是否执行）。
        
        Args:
            line: 待分词的文本
            
        Returns:
            str: 分词结果或原始文本（Infinity模式）
        """
        # 延迟导入，避免和全局 settings 初始化阶段形成循环依赖
        from common import settings
        if settings.DOC_ENGINE_INFINITY:
            return line  # Infinity 引擎内置分词，直接返回
        else:
            return super().tokenize(line)

    def fine_grained_tokenize(self, tks: str) -> str:
        """细粒度分词处理（根据文档引擎类型决定是否执行）。
        
        Args:
            tks: 待分词的文本
            
        Returns:
            str: 细粒度分词结果或原始文本（Infinity模式）
        """
        # 延迟导入，避免和全局 settings 初始化阶段形成循环依赖
        from common import settings
        if settings.DOC_ENGINE_INFINITY:
            return tks  # Infinity 引擎内置分词，直接返回
        else:
            return super().fine_grained_tokenize(tks)


def is_chinese(s):
    """判断文本是否主要为中文。
    
    Args:
        s: 待检测文本
        
    Returns:
        bool: 是否为中文
    """
    return infinity.rag_tokenizer.is_chinese(s)


def is_number(s):
    """判断文本是否为数字串。
    
    Args:
        s: 待检测文本
        
    Returns:
        bool: 是否为数字串
    """
    return infinity.rag_tokenizer.is_number(s)


def is_alphabet(s):
    """判断文本是否为字母串。
    
    Args:
        s: 待检测文本
        
    Returns:
        bool: 是否为字母串
    """
    return infinity.rag_tokenizer.is_alphabet(s)


def naive_qie(txt):
    """执行底层 tokenizer 的朴素切分（不做复杂处理）。
    
    Args:
        txt: 待切分文本
        
    Returns:
        list: 切分后的token列表
    """
    return infinity.rag_tokenizer.naive_qie(txt)


# 全局实例和快捷函数
tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize                    # 分词函数
fine_grained_tokenize = tokenizer.fine_grained_tokenize  # 细粒度分词函数
tag = tokenizer.tag                              # 词性标注函数
freq = tokenizer.freq                            # 词频查询函数
tradi2simp = tokenizer._tradi2simp               # 繁体转简体函数
strQ2B = tokenizer._strQ2B                       # 全角转半角函数