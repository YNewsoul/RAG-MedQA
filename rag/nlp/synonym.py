"""同义词查询器。

优先读取本地词典（包括通用同义词和医疗领域同义词），
其次在纯英文 token 场景下回退到 WordNet，
用于增强全文检索时的召回能力。

检索流程：
1. 查询本地同义词词典（synonym.json + medical_synonym.json）
2. 如果本地未命中且是纯英文单词，查询 WordNet
3. 返回同义词列表（最多topn个）
"""

import logging
import json
import os
import time
import re

from nltk.corpus import wordnet

from common.file_utils import get_project_base_directory


# 进程启动时同步加载 WordNet，避免并发请求触发懒加载竞态
try:
    wordnet.ensure_loaded()
except Exception:
    logging.warning("Fail to load wordnet.ensure_loaded()")


class Dealer:
    """同义词查询器类。
    
    支持本地词典查询和 WordNet 回退，用于增强检索召回能力。
    """
    def __init__(self, redis=None):
        """初始化同义词查询器。
        
        Args:
            redis: Redis连接（用于实时更新同义词词典）
        """
        self.lookup_num = 100000000  # 查询计数（用于触发词典重载）
        self.load_tm = time.time() - 1000000  # 上次加载时间
        self.dictionary = None  # 同义词词典
        
        # 加载通用同义词词典
        path = os.path.join(get_project_base_directory(), "rag/res", "synonym.json")
        try:
            with open(path, "r") as f:
                self.dictionary = json.load(f)
            # 统一转为小写
            self.dictionary = {
                (k.lower() if isinstance(k, str) else k): v
                for k, v in self.dictionary.items()
            }
        except Exception:
            logging.warning("Missing synonym.json")
            self.dictionary = {}

        # 合并医疗领域同义词词典；若有冲突则以医疗词典为准
        med_path = os.path.join(
            get_project_base_directory(),
            "rag/res",
            "medical_synonym.json",
        )
        try:
            with open(med_path, "r", encoding="utf-8") as f:
                med_dict = json.load(f)
            for k, v in med_dict.items():
                self.dictionary[k.lower() if isinstance(k, str) else k] = v
            logging.info(f"Loaded {len(med_dict)} medical synonym groups.")
        except FileNotFoundError:
            logging.warning("medical_synonym.json not found, skipping medical synonyms.")
        except Exception as e:
            logging.warning(f"Failed to load medical_synonym.json: {e}")

        # 检查Redis连接（用于实时同义词更新）
        if not redis:
            logging.warning("Realtime synonym is disabled, since no redis connection.")
        if not len(self.dictionary.keys()):
            logging.warning("Fail to load synonym")

        self.redis = redis
        self.load()

    def load(self):
        """从Redis重载同义词词典（定时更新机制）。
        
        更新条件：
        1. 必须有Redis连接
        2. 查询次数超过100次
        3. 距离上次加载超过1小时
        """
        if not self.redis:
            return

        if self.lookup_num < 100:
            return
        tm = time.time()
        if tm - self.load_tm < 3600:  # 1小时内不重复加载
            return

        self.load_tm = time.time()
        self.lookup_num = 0
        d = self.redis.get("kevin_synonyms")
        if not d:
            return
        try:
            d = json.loads(d)
            self.dictionary = d
        except Exception as e:
            logging.error("Fail to load synonym!" + str(e))

    def lookup(self, tk, topn=8):
        """查询同义词。
        
        查询流程：
        1. 先查本地词典（通用+医疗领域）
        2. 如果本地未命中且是纯英文单词，查询 WordNet
        3. 返回最多topn个同义词
        
        Args:
            tk: 待查询的token
            topn: 最多返回的同义词数量（默认8）
            
        Returns:
            list: 同义词列表
        """
        if not tk or not isinstance(tk, str):
            return []

        # 1. 先查本地词典；这里的 key 和 tk 都已统一成小写
        self.lookup_num += 1
        self.load()
        key = re.sub(r"[ \t]+", " ", tk.strip())
        res = self.dictionary.get(key, [])
        if isinstance(res, str):
            res = [res]
        if res:
            return res[:topn]

        # 2. 本地词典未命中且 token 为纯英文时，回退到 WordNet
        if re.fullmatch(r"[a-z]+", tk):
            wn_set = {
                re.sub("_", " ", syn.name().split(".")[0])
                for syn in wordnet.synsets(tk)
            }
            wn_set.discard(tk)  # 去掉原始 token 本身
            wn_res = [t for t in wn_set if t]
            return wn_res[:topn]

        # 3. 两个来源都没命中时返回空列表
        return []


if __name__ == "__main__":
    dl = Dealer()
    print(dl.dictionary)