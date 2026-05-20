"""全文查询构造器模块。

这是 RAG 系统中的查询处理核心模块，负责将用户自然语言问题转换为
全文检索引擎可理解的查询表达式，实现精准的文档召回。

核心功能模块：
┌─────────────────────────────────────────────────────────────┐
│ 1. 文本预处理                                               │
│    - 中英文混合处理（自动添加空格分隔）                       │
│    - 繁简转换、全角半角转换                                 │
│    - 特殊字符清理与规范化                                   │
├─────────────────────────────────────────────────────────────┤
│ 2. 关键词提取与权重计算                                     │
│    - 基于词频和TF-IDF的权重计算                             │
│    - 细粒度分词处理                                         │
│    - 关键词数量控制（最多32个）                              │
├─────────────────────────────────────────────────────────────┤
│ 3. 同义词扩展                                               │
│    - 中英文同义词查询                                       │
│    - 同义词加权（权重为原词的1/4）                          │
├─────────────────────────────────────────────────────────────┤
│ 4. 查询表达式构建                                           │
│    - 多字段加权查询（标题、内容、关键词等）                  │
│    - 短语查询与邻近查询                                     │
│    - Boolean OR 组合多个查询条件                            │
├─────────────────────────────────────────────────────────────┤
│ 5. 混合相似度计算                                          │
│    - 向量余弦相似度                                         │
│    - Token 匹配相似度                                       │
│    - 线性组合两种相似度                                     │
└─────────────────────────────────────────────────────────────┘
"""

import logging
import json
import re
from collections import defaultdict

from common.query_base import QueryBase
from common.doc_store.doc_store_base import MatchTextExpr
from rag.nlp import rag_tokenizer, term_weight, synonym


class FulltextQueryer(QueryBase):
    """全文查询构造器类。
    
    继承自 QueryBase，负责将用户自然语言问题转换为全文检索引擎可理解的
    查询表达式。支持中英文混合查询、同义词扩展、短语加权等高级功能。
    """

    def __init__(self):
        """初始化查询器。

        创建词权重计算器和同义词处理器实例，并配置查询字段列表。
        """
        self.tw = term_weight.Dealer()  # 词权重计算器（基于TF-IDF）
        self.syn = synonym.Dealer()     # 同义词处理器

        # ========== 查询字段配置 ==========
        # 格式: "字段名^权重"，数值越大优先级越高
        # 权重设计原则：
        # - 重要关键词 > 标题 > 问题 > 内容
        # - 细粒度分词权重低于普通分词
        self.query_fields = [
            "title_tks^10",      # 标题分词（权重10）- 标题是文档核心
            "title_sm_tks^5",    # 标题细粒度分词（权重5）
            "important_kwd^30",  # 重要关键词（权重30）- 最高优先级
            "important_tks^20",  # 重要分词（权重20）
            "question_tks^20",   # 问题分词（权重20）- QA场景专用
            "content_ltks^2",    # 内容分词（权重2）- 较低优先级
            "content_sm_ltks",   # 内容细粒度分词（权重1）- 最低优先级
        ]

    def question(self, txt, tbl="qa", min_match: float = 0.6):
        """将用户问题转换为全文检索查询表达式。

        核心处理流程：
        1. 文本预处理（中英文分隔、繁简转换、特殊字符清理）
        2. 判断语言类型（中文/非中文）
        3. 分词与权重计算
        4. 同义词扩展
        5. 构建查询表达式

        Args:
            txt (str): 用户输入的问题
            tbl (str): 查询表名（默认 "qa"，即问答表）
            min_match (float): 最小匹配比例（默认 0.6，即至少匹配60%的关键词）
            
        Returns:
            tuple: (MatchTextExpr对象, 关键词列表)
        """
        original_query = txt  # 保存原始查询用于记录

        # ========== 步骤1: 文本预处理 ==========
        # 在中英文之间添加空格，便于后续分词处理
        txt = self.add_space_between_eng_zh(txt)

        # 清理特殊字符（Infinity 检索引擎的保留字符）
        # 处理流程：
        # 1. 转小写
        # 2. 全角转半角（strQ2B）
        # 3. 繁体转简体（tradi2simp）
        # 4. 替换特殊字符为空格
        # 使用原始字符串和正确的转义
        special_chars_pattern = r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>*~'\"\\]+"
        txt = re.sub(
            special_chars_pattern,
            " ",
            rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower())),
        ).strip()

        otxt = txt  # 保存清理后的文本作为备选
        txt = self.rmWWW(txt)  # 移除可能的 WWW 前缀

        # ========== 步骤2: 判断语言类型并分支处理 ==========
        if not self.is_chinese(txt):
            # ========== 英文/非中文处理分支 ==========
            txt = self.rmWWW(txt)
            
            # 分词处理
            tks = rag_tokenizer.tokenize(txt).split()
            keywords = [t for t in tks if t]

            # 计算词权重（不进行预处理）
            tks_w = self.tw.weights(tks, preprocess=False)
            tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w]
            tks_w = [(re.sub(r"^[\+-]", "", tk), w) for tk, w in tks_w if tk]
            tks_w = [(tk.strip(), w) for tk, w in tks_w if tk.strip()]

            # 同义词扩展（最多处理前256个词）
            syns = []
            for tk, w in tks_w[:256]:
                # 去掉同义词里的单引号，避免 Infinity 词法解析报错
                syn = [rag_tokenizer.tokenize(s).replace("'", "") for s in self.syn.lookup(tk)]
                keywords.extend(syn)
                # 同义词权重为原词的1/4（降低权重，避免干扰主关键词）
                syn = ["\"{}\"^{:.4f}".format(s, w / 4.) for s in syn if s.strip()]
                syns.append(" ".join(syn))

            # 构建查询表达式
            q = []
            for (tk, w), syn in zip(tks_w, syns):
                if tk and not re.match(r"[.^+\\(\\)-]", tk):
                    # 格式: (关键词^权重 同义词表达式)
                    q.append("({}^{:.4f} {})".format(tk, w, syn))
            # 添加双词短语查询（权重加倍，提升短语匹配优先级）
            for i in range(1, len(tks_w)):
                left, right = tks_w[i - 1][0].strip(), tks_w[i][0].strip()
                if not left or not right:
                    continue
                # 格式: "词1 词2"^权重（权重为两个词的最大值*2）
                q.append(
                    '"%s %s"^%.4f'
                    % (
                        tks_w[i - 1][0],
                        tks_w[i][0],
                        max(tks_w[i - 1][1], tks_w[i][1]) * 2,
                    )
                )

            # 如果没有有效查询词，使用原始文本作为备选
            if not q:
                q.append(txt)

            query = " ".join(q)

            return MatchTextExpr(
                self.query_fields, query, 100, {"original_query": original_query}
            ), keywords

        # ========== 中文处理分支 ==========
        # 定义细粒度分词判断函数
        def need_fine_grained_tokenize(tk):
            """判断是否需要细粒度分词。

            条件：
            1. 长度 >= 3
            2. 不是纯数字/字母组合

            Args:
                tk (str): 待判断的词

            Returns:
                bool: 是否需要细粒度分词
            """
            """判断是否需要细粒度分词。

            细粒度分词用于处理较长的中文词汇，将其拆分为更小的语义单元。
            例如："冠状动脉粥样硬化" -> ["冠状", "动脉", "粥样", "硬化"]

            判断条件：
            1. 长度 >= 3（太短的词不需要拆分）
            2. 不是纯数字/字母组合（如 "COVID-19" 不需要拆分）

            Args:
                tk (str): 待判断的词

            Returns:
                bool: True 表示需要细粒度分词，False 表示不需要
            """
            if len(tk) < 3:
                return False
            if re.match(r"[0-9a-z\.\+#_\*-]+$", tk):
                return False
            return True

        txt = self.rmWWW(txt)
        qs, keywords = [], []
        for tt in self.tw.split(txt)[:256]:  # .split():
            if not tt:
                continue
            keywords.append(tt)
            twts = self.tw.weights([tt])
            syns = self.syn.lookup(tt)
            if syns and len(keywords) < 32:
                keywords.extend(syns)
            logging.debug(json.dumps(twts, ensure_ascii=False))
            tms = []
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                sm = (
                    rag_tokenizer.fine_grained_tokenize(tk).split()
                    if need_fine_grained_tokenize(tk)
                    else []
                )
                sm = [
                    re.sub(
                        r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-]+",
                        "",
                        m,
                    )
                    for m in sm
                ]
                sm = [self.sub_special_char(m) for m in sm if len(m) > 1]
                sm = [m for m in sm if len(m) > 1]

                if len(keywords) < 32:
                    keywords.append(re.sub(r"[ \\\"']+", "", tk))
                    keywords.extend(sm)

                tk_syns = self.syn.lookup(tk)
                tk_syns = [self.sub_special_char(s) for s in tk_syns]
                if len(keywords) < 32:
                    keywords.extend([s for s in tk_syns if s])
                tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
                tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]

                if len(keywords) >= 32:
                    break

                tk = self.sub_special_char(tk)
                if tk.find(" ") > 0:
                    tk = '"%s"' % tk
                if tk_syns:
                    tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
                if sm:
                    tk = f'{tk} OR "%s" OR ("%s"~2)^0.5' % (" ".join(sm), " ".join(sm))
                if tk.strip():
                    tms.append((tk, w))

            tms = " ".join([f"({t})^{w}" for t, w in tms])

            if len(twts) > 1:
                tms += ' ("%s"~2)^1.5' % rag_tokenizer.tokenize(tt)

            syns = " OR ".join(
                [
                    '"%s"'
                    % rag_tokenizer.tokenize(self.sub_special_char(s))
                    for s in syns
                ]
            )
            if syns and tms:
                tms = f"({tms})^5 OR ({syns})^0.7"

            qs.append(tms)

        if qs:
            query = " OR ".join([f"({t})" for t in qs if t])
            if not query:
                query = otxt
            return MatchTextExpr(
                self.query_fields, query, 100, {"minimum_should_match": min_match, "original_query": original_query}
            ), keywords
        return None, keywords

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
        """计算混合相似度（向量相似度 + Token相似度的线性组合）。

        混合相似度 = 向量余弦相似度 * vtweight + Token相似度 * tkweight

        设计目的：
        - 向量相似度：捕捉语义相似度，擅长处理同义词和语义相关的内容
        - Token相似度：捕捉字面匹配，擅长处理精确关键词匹配
        - 组合后可以兼顾语义理解和精确匹配，提升检索准确性

        权重设置原则（默认）：
        - vtweight=0.7：向量相似度权重更高，因为语义理解更重要
        - tkweight=0.3：Token相似度作为补充

        Args:
            avec (numpy.ndarray): 查询向量（一维数组）
            bvecs (numpy.ndarray): 待比较的向量列表（二维数组，每行一个向量）
            atks (str/list): 查询关键词（字符串或列表）
            btkss (list): 待比较的关键词列表（多个文档的关键词）
            tkweight (float): Token相似度权重（默认0.3）
            vtweight (float): 向量相似度权重（默认0.7）
            
        Returns:
            tuple: (混合相似度数组, token相似度列表, 向量相似度数组)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # ========== 步骤1: 计算向量余弦相似度 ==========
        # cosine_similarity 返回形状为 (1, n) 的数组，n 为待比较向量数量
        sims = cosine_similarity([avec], bvecs)

        # ========== 步骤2: 计算Token相似度 ==========
        tksim = self.token_similarity(atks, btkss)

        # ========== 步骤3: 处理边界情况 ==========
        # 如果向量相似度全为0（可能是向量未正确初始化），只返回Token相似度
        if np.sum(sims[0]) == 0:
            return np.array(tksim), tksim, sims[0]

        # ========== 步骤4: 线性组合两种相似度 ==========
        hybrid_sim = np.array(sims[0]) * vtweight + np.array(tksim) * tkweight

        return hybrid_sim, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        """计算查询关键词与多个文档关键词之间的相似度。

        相似度计算采用带权重的词匹配策略：
        1. 单个词匹配权重占40%
        2. 双词短语匹配权重占60%

        这样设计的目的是：
        - 鼓励短语匹配（更精确）
        - 同时保留单个词匹配（更灵活）

        Args:
            atks (str/list): 查询关键词（字符串或列表）
            btkss (list): 待比较的关键词列表（多个文档）
            
        Returns:
            list: 每个文档的Token相似度（0~1之间）
        """
        def to_dict(tks):
            """将关键词列表转换为带权重的字典。

            处理逻辑：
            1. 如果输入是字符串，先按空格分词
            2. 计算每个词的权重
            3. 单个词权重 * 0.4
            4. 双词短语权重 * 0.6（取两个词的最大权重）

            Args:
                tks (str/list): 关键词

            Returns:
                defaultdict: 词/短语到权重的映射
            """
            if isinstance(tks, str):
                tks = tks.split()
            d = defaultdict(int)
            wts = self.tw.weights(tks, preprocess=False)
            for i, (t, c) in enumerate(wts):
                d[t] += c * 0.4  # 单个词权重占40%
                if i + 1 < len(wts):
                    _t, _c = wts[i + 1]
                    d[t + _t] += max(c, _c) * 0.6  # 双词短语权重占60%
            return d

        # 将查询关键词和文档关键词都转换为权重字典
        atks = to_dict(atks)
        btkss = [to_dict(tks) for tks in btkss]

        # 计算每个文档与查询的相似度
        return [self.similarity(atks, btks) for btks in btkss]

    def similarity(self, qtwt, dtwt):
        """计算两个关键词权重字典的相似度。

        相似度计算公式：
        similarity = (查询词在文档中匹配的权重之和) / (查询词总权重)

        设计特点：
        - 使用 1e-9 作为分母的最小值，避免除零错误
        - 归一化到 [0, 1] 区间
        - 支持字符串输入（自动转换为权重字典）

        Args:
            qtwt (dict/str): 查询关键词权重字典或字符串
            dtwt (dict/str): 文档关键词权重字典或字符串
            
        Returns:
            float: 相似度分数（0~1之间）
        """
        # ========== 输入预处理 ==========
        # 如果输入是字符串，先转换为权重字典
        if isinstance(dtwt, str):
            dtwt = {t: w for t, w in self.tw.weights(self.tw.split(dtwt), preprocess=False)}
        if isinstance(qtwt, str):
            qtwt = {t: w for t, w in self.tw.weights(self.tw.split(qtwt), preprocess=False)}

        # ========== 计算匹配权重之和 ==========
        # 初始值设为 1e-9 避免除零错误
        match_weight = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                match_weight += v

        # ========== 计算查询词总权重 ==========
        total_weight = 1e-9
        for k, v in qtwt.items():
            total_weight += v

        # ========== 返回归一化相似度 ==========
        return match_weight / total_weight

    def paragraph(self, content_tks: str, keywords: list = [], keywords_topn=30):
        """从段落内容构建查询表达式。

        与 question() 方法的区别：
        - question(): 处理用户问题，强调精准匹配和同义词扩展
        - paragraph(): 处理文档段落，强调从内容中提取关键词

        应用场景：
        - 当需要根据一段文本生成检索查询时使用
        - 可用于文档摘要检索、相关文档推荐等场景

        Args:
            content_tks (str/list): 内容分词字符串或列表
            keywords (list): 已有关键词列表（可选）
            keywords_topn (int): 最多提取的关键词数量（默认30）
            
        Returns:
            MatchTextExpr: 查询表达式对象
        """
        # ========== 输入预处理 ==========
        if isinstance(content_tks, str):
            # 将字符串转换为字符列表（这里可能是代码意图问题，应该是分词列表）
            content_tks = [c.strip() for c in content_tks.strip() if c.strip()]

        # ========== 计算词权重 ==========
        tks_w = self.tw.weights(content_tks, preprocess=False)

        # 保存原始关键词用于记录
        origin_keywords = keywords.copy()
        # 将已有关键词添加引号（作为短语查询）
        keywords = [f'"{k.strip()}"' for k in keywords]

        # ========== 提取topn关键词并构建查询表达式 ==========
        # 按权重降序排列，取前 keywords_topn 个
        for tk, w in sorted(tks_w, key=lambda x: x[1] * -1)[:keywords_topn]:
            # 获取同义词
            tk_syns = self.syn.lookup(tk)
            tk_syns = [self.sub_special_char(s) for s in tk_syns]
            tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
            tk_syns = [f'"{s}"' if s.find(" ") > 0 else s for s in tk_syns]

            # 处理当前词
            tk = self.sub_special_char(tk)
            if tk.find(" ") > 0:
                tk = '"%s"' % tk

            # 添加同义词（权重0.2）
            if tk_syns:
                tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)

            # 添加到关键词列表
            if tk:
                keywords.append(f"{tk}^{w}")

        # ========== 构建最终查询表达式 ==========
        # 计算最小匹配数：至少匹配3个或关键词总数的10%（取较小值）
        min_match = min(3, round(len(keywords) / 10))

        return MatchTextExpr(
            self.query_fields, 
            " ".join(keywords), 
            100,
            {
                "minimum_should_match": min_match,
                "original_query": " ".join(origin_keywords)
            }
        )