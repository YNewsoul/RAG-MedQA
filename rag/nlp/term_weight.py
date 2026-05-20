"""词项权重计算器。

这个模块负责：
1. 预处理原始文本，去掉停用词和噪声 token
2. 根据词频(TF)、文档频率(DF)、命名实体识别(NER)类型、词性等信息给 token 赋权
3. 为查询构造与相似度计算提供更稳定的关键词权重基础

权重计算公式:
weight = (0.3 * IDF(word_freq) + 0.7 * IDF(doc_freq)) * NER_weight * POS_weight
"""

import logging
import math
import json
import re
import os
import numpy as np
from rag.nlp import rag_tokenizer
from common.file_utils import get_project_base_directory


class Dealer:
    """关键词权重计算器类。
    
    根据词频、文档频率、命名实体类型和词性等因素计算词的权重，
    用于提升检索和相似度计算的准确性。
    """
    def __init__(self):
        # 停用词表主要用于中文问答场景，避免虚词对检索造成过大干扰。
        self.stop_words = set(["请问",
                               "您",
                               "你",
                               "我",
                               "他",
                               "是",
                               "的",
                               "就",
                               "有",
                               "于",
                               "及",
                               "即",
                               "在",
                               "为",
                               "最",
                               "有",
                               "从",
                               "以",
                               "了",
                               "将",
                               "与",
                               "吗",
                               "吧",
                               "中",
                               "#",
                               "什么",
                               "怎么",
                               "哪个",
                               "哪些",
                               "啥",
                               "相关"])

        def load_dict(fnm):
            """读取词频/文档频率词典。"""
            res = {}
            with open(fnm, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    arr = line.replace("\n", "").split("\t")
                    if len(arr) < 2:
                        res[arr[0]] = 0
                    else:
                        res[arr[0]] = int(arr[1])

            c = 0
            for _, v in res.items():
                c += v
            if c == 0:
                return set(res.keys())
            return res

        fnm = os.path.join(get_project_base_directory(), "rag/res")
        self.ne, self.df = {}, {}
        try:
            with open(os.path.join(fnm, "ner.json"), "r") as f:
                self.ne = json.load(f)
        except Exception:
            logging.warning("Load ner.json FAIL!")
        try:
            self.df = load_dict(os.path.join(fnm, "term.freq"))
        except Exception:
            logging.warning("Load term.freq FAIL!")

    def pretoken(self, txt, num=False, stpwd=True):
        """对原始文本做预分词、停用词过滤和噪声剔除。"""
        patt = [
            r"[~—\t @#%!<>,\.\?\":;'\{\}\[\]_=\(\)\|，。？》•●○↓《；‘’：“”【¥ 】…￥！、·（）×`&\\/「」\\]"
        ]
        rewt = [
        ]
        for p, r in rewt:
            txt = re.sub(p, r, txt)

        res = []
        for t in rag_tokenizer.tokenize(txt).split():
            tk = t
            if (stpwd and tk in self.stop_words) or (
                    re.match(r"[0-9]$", tk) and not num):
                continue
            for p in patt:
                if re.match(p, t):
                    tk = "#"
                    break
            # tk = re.sub(r"([\+\\-])", r"\\\1", tk)
            if tk != "#" and tk:
                res.append(tk)
        return res

    def token_merge(self, tks):
        """把连续的短 token 合并，减少切分过碎的问题。"""
        def one_term(t):
            return len(t) == 1 or re.match(r"[0-9a-z]{1,2}$", t)

        res, i = [], 0
        while i < len(tks):
            j = i
            if i == 0 and one_term(tks[i]) and len(
                    tks) > 1 and (len(tks[i + 1]) > 1 and not re.match(r"[0-9a-zA-Z]", tks[i + 1])):  # 多 工位
                res.append(" ".join(tks[0:2]))
                i = 2
                continue

            while j < len(
                    tks) and tks[j] and tks[j] not in self.stop_words and one_term(tks[j]):
                j += 1
            if j - i > 1:
                if j - i < 5:
                    res.append(" ".join(tks[i:j]))
                    i = j
                else:
                    res.append(" ".join(tks[i:i + 2]))
                    i = i + 2
            else:
                if len(tks[i]) > 0:
                    res.append(tks[i])
                i += 1
        return [t for t in res if t]

    def ner(self, t):
        """查询 token 的命名实体类型。
        
        Args:
            t: token字符串
            
        Returns:
            str: NER类型（如toxic, func, corp, loca等），未找到返回空字符串
        """
        if not self.ne:
            return ""
        res = self.ne.get(t, "")
        if res:
            return res

    def split(self, txt):
        """按空白拆分，并尝试把相邻英文片段合并回短语。
        
        Args:
            txt: 文本字符串
            
        Returns:
            list: 拆分后的token列表（相邻英文会合并）
        """
        tks = []
        for t in re.sub(r"[ \t]+", " ", txt).split():
            # 如果前一个token和当前token都是英文，且不是函数名，则合并
            if tks and re.match(r".*[a-zA-Z]$", tks[-1]) and \
                    re.match(r".*[a-zA-Z]$", t) and tks and \
                    self.ne.get(t, "") != "func" and self.ne.get(tks[-1], "") != "func":
                tks[-1] = tks[-1] + " " + t
            else:
                tks.append(t)
        return tks

    def weights(self, tks, preprocess=True):
        """计算 token 权重。
        
        权重会综合考虑以下因素：
        1. 词频(TF) - 通过IDF计算
        2. 文档频率(DF) - 通过IDF计算
        3. 命名实体类型(NER) - 如公司、地点、学校等
        4. 词性(POS) - 名词权重更高，虚词权重更低
        
        最终权重公式:
        weight = (0.3 * IDF(word_freq) + 0.7 * IDF(doc_freq)) * NER_weight * POS_weight
        
        Args:
            tks: token列表
            preprocess: 是否需要预处理（分词、合并等）
            
        Returns:
            list: [(token, weight), ...] 归一化后的权重列表
        """
        num_pattern = re.compile(r"[0-9,.]{2,}$")          # 数字模式
        short_letter_pattern = re.compile(r"[a-z]{1,2}$")   # 短英文模式
        num_space_pattern = re.compile(r"[0-9. -]{2,}$")    # 数字空格模式
        letter_pattern = re.compile(r"[a-z. -]+$")          # 英文模式

        def ner(t):
            """获取NER类型权重。"""
            if num_pattern.match(t):
                return 2  # 数字权重2
            if short_letter_pattern.match(t):
                return 0.01  # 短英文权重极低
            if not self.ne or t not in self.ne:
                return 1  # 默认权重1
            # NER类型权重映射
            m = {"toxic": 2, "func": 1, "corp": 3, "loca": 3, "sch": 3, "stock": 3,
                 "firstnm": 1}
            return m[self.ne[t]]

        def postag(t):
            """获取词性权重。"""
            t = rag_tokenizer.tag(t)
            if t in set(["r", "c", "d"]):
                return 0.3  # 代词、连词、副词权重低
            if t in set(["ns", "nt"]):
                return 3   # 地名、机构名权重高
            if t in set(["n"]):
                return 2   # 名词权重较高
            if re.match(r"[0-9-]+", t):
                return 2   # 数字权重较高
            return 1

        def freq(t):
            """获取词频。"""
            if num_space_pattern.match(t):
                return 3
            s = rag_tokenizer.freq(t)
            if not s and letter_pattern.match(t):
                return 300
            if not s:
                s = 0

            # 如果词频为0且词较长，尝试细粒度分词后计算
            if not s and len(t) >= 4:
                s = [tt for tt in rag_tokenizer.fine_grained_tokenize(t).split() if len(tt) > 1]
                if len(s) > 1:
                    s = np.min([freq(tt) for tt in s]) / 6.
                else:
                    s = 0

            return max(s, 10)

        def df(t):
            """获取文档频率。"""
            if num_space_pattern.match(t):
                return 5
            if t in self.df:
                return self.df[t] + 3
            elif letter_pattern.match(t):
                return 300
            elif len(t) >= 4:
                s = [tt for tt in rag_tokenizer.fine_grained_tokenize(t).split() if len(tt) > 1]
                if len(s) > 1:
                    return max(3, np.min([df(tt) for tt in s]) / 6.)

            return 3

        def idf(s, N):
            """计算逆文档频率。"""
            return math.log10(10 + ((N - s + 0.5) / (s + 0.5)))

        tw = []
        if not preprocess:
            # 不预处理：直接计算权重
            idf1 = np.array([idf(freq(t), 10000000) for t in tks])
            idf2 = np.array([idf(df(t), 1000000000) for t in tks])
            wts = (0.3 * idf1 + 0.7 * idf2) * np.array([ner(t) * postag(t) for t in tks])
            wts = [s for s in wts]
            tw = list(zip(tks, wts))
        else:
            # 预处理：先分词合并，再计算权重
            for tk in tks:
                tt = self.token_merge(self.pretoken(tk, True))
                idf1 = np.array([idf(freq(t), 10000000) for t in tt])
                idf2 = np.array([idf(df(t), 1000000000) for t in tt])
                wts = (0.3 * idf1 + 0.7 * idf2) * np.array([ner(t) * postag(t) for t in tt])
                wts = [s for s in wts]
                tw.extend(zip(tt, wts))

        # 归一化权重
        S = np.sum([s for _, s in tw])
        return [(t, s / S) for t, s in tw]