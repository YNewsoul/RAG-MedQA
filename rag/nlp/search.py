"""检索器实现。

这里封装了项目里的核心检索逻辑，是理解 RAG 检索层的关键文件。

核心功能包括：
1. **全文检索**：基于关键词的倒排索引检索
2. **向量召回**：基于语义相似度的向量检索
3. **混合检索**：全文分数 + 向量分数的融合排序
4. **重排序**：使用重排模型对初步结果二次排序
5. **TOC 增强**：基于文档目录结构的检索增强
6. **父子 chunk 回收**：利用文档结构关系扩展召回
7. **引用标注**：自动为大模型回答添加引用来源

检索流程：
1. 用户提问 → 生成查询向量和全文查询表达式
2. 执行混合检索（全文 + 向量）
3. 对结果进行重排序
4. 利用 TOC 和父子关系扩展召回
5. 为回答自动添加引用标注
"""

import json
import logging
import re
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from rag.nlp import rag_tokenizer, query
import numpy as np
from common.doc_store.doc_store_base import MatchDenseExpr, FusionExpr, OrderByExpr, DocStoreConnection
from common.string_utils import remove_redundant_spaces
from common.float_utils import get_float
from common.constants import PAGERANK_FLD, TAG_FLD
from common import settings

from common.misc_utils import thread_pool_exec

from common.constants import SYSTEM_TENANT_ID, SYSTEM_INDEX_NAME


def index_name(uid=None):
    """获取索引名称。
    
    目前所有用户共享同一个系统索引。
    
    Args:
        uid: 用户ID（暂未使用）
        
    Returns:
        str: 索引名称
    """
    # 这里虽然保留了 `uid` 参数，但当前分支实际上已经统一收敛到系统级共享索引。
    # 也就是说，索引层面不再做用户隔离，真正的检索范围控制主要依赖 kb_id / doc_id 等过滤条件。
    return SYSTEM_INDEX_NAME


class Dealer:
    """检索器核心类，封装所有检索相关操作。
    
    该类是RAG系统的检索中枢，负责：
    1. 将用户问题转换为检索表达式
    2. 执行混合检索（全文+向量）
    3. 对候选结果进行重排序
    4. 添加引用标注
    5. TOC增强和父子chunk处理
    
    Args:
        dataStore: 文档存储连接对象（如Elasticsearch连接）
    """
    def __init__(self, dataStore: DocStoreConnection):
        # `qryr` 负责“把问题变成可检索表达式”，以及后续 token 相似度相关计算。
        # 可以把它理解成“检索前处理 + 规则相似度工具箱”。
        self.qryr = query.FulltextQueryer()
        # `dataStore` 是对底层存储引擎的统一抽象。
        # 当前文件并不直接依赖某个具体 ES / Infinity SDK，而是通过它执行 search / get / sql 等操作。
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        """检索结果数据结构。
        
        Attributes:
            total: 总结果数
            ids: 文档块ID列表
            query_vector: 查询向量（用于后续相似度计算）
            field: 文档字段内容字典
            highlight: 高亮关键词信息
            aggregation: 聚合统计信息（如按文档名称分组）
            keywords: 从查询中提取的关键词
            group_docs: 分组文档列表
        """
        total: int                                  # 总结果数
        ids: list[str]                              # 文档块ID列表
        query_vector: list[float] | None = None     # 查询向量
        field: dict | None = None                   # 文档字段内容
        highlight: dict | None = None               # 高亮关键词
        aggregation: list | dict | None = None      # 聚合统计
        keywords: list[str] | None = None           # 提取的关键词
        group_docs: list[list] | None = None        # 分组文档

    async def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        """把问题编码成向量检索表达式。
        
        将文本问题通过嵌入模型转换为向量，并构建向量检索表达式。
        
        Args:
            txt: 查询文本
            emb_mdl: 嵌入模型实例
            topk: 返回结果数量
            similarity: 相似度阈值
            
        Returns:
            MatchDenseExpr: 向量检索表达式对象
            
        Raises:
            Exception: 向量维度不符合预期时抛出异常
        """
        # 许多 embedding SDK 仍然是同步阻塞接口，这里放进线程池执行，
        # 可以避免在异步请求链路里直接卡住事件循环。
        qv, _ = await thread_pool_exec(emb_mdl.encode_queries, txt)
        
        # 查询向量按设计应该是一维数组；如果出现二维结构，说明模型封装层返回格式异常。
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        
        # 统一把元素转换成 float，避免底层引擎收到字符串等不稳定类型。
        embedding_data = [get_float(v) for v in qv]
        
        # 向量字段名和维度绑定，例如 `q_768_vec`、`q_1024_vec`。
        # 这样底层索引可以同时支持多种维度的向量字段，但查询时必须使用正确列名。
        vector_column_name = f"q_{len(embedding_data)}_vec"
        
        # 这里返回的不是“向量数组本身”，而是底层文档引擎可执行的 dense match 表达式。
        # 后续它会和全文检索表达式一起交给 `dataStore.search` 做混合召回。
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

    def get_filters(self, req):
        """生成检索过滤条件。
        
        从请求参数中提取过滤条件，限制检索范围。
        
        Args:
            req: 请求字典，包含过滤参数
            
        Returns:
            dict: 过滤条件字典
            
        支持的过滤参数：
            - kb_ids: 知识库ID列表
            - doc_ids: 文档ID列表
            - knowledge_graph_kwd: 知识图谱关键词
            - available_int: 可用状态
            - entity_kwd: 实体关键词
            - from_entity_kwd: 起始实体
            - to_entity_kwd: 目标实体
            - removed_kwd: 是否已删除
        """
        # 这个函数只做一件事：把请求里“限制检索范围”的部分摘出来。
        # 例如只搜某些知识库、某些文档，或者限定图谱/实体相关字段。
        condition = dict()
        
        # 知识库和文档过滤是最常见、也最关键的两类范围约束。
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        
        # 其余过滤项主要服务于图谱检索、状态控制和特殊业务筛选。
        # TODO(yzc): `available_int` 允许为空，但 Infinity 暂不支持可空列
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", 
                    "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        
        return condition

    async def search(self, req, idx_names: str | list[str],
               kb_ids: list[str],
               emb_mdl=None,
               highlight: bool | list | None = None,
               rank_feature: dict | None = None
               ):
        """执行一次底层检索，并返回统一的搜索结果对象。
        
        这是核心检索函数，支持全文检索、向量检索和混合检索三种模式。
        
        Args:
            req: 请求字典，包含查询参数
            idx_names: 索引名称（单个或多个）
            kb_ids: 知识库ID列表
            emb_mdl: 嵌入模型实例（为None时仅执行全文检索）
            highlight: 是否高亮（True/False/字段列表）
            rank_feature: 排序特征配置
            
        Returns:
            SearchResult: 检索结果对象
            
        检索流程：
        1. 解析请求参数（分页、字段、过滤条件）
        2. 如果没有问题，执行简单排序检索
        3. 如果有问题：
           - 仅全文检索：生成全文查询表达式
           - 混合检索：生成全文表达式 + 向量表达式 + 融合表达式
        4. 如果首次检索无结果，放宽条件重试
        5. 提取关键词、高亮、聚合信息
        6. 返回统一的SearchResult对象
        """
        # `highlight=None` 时显式退化成 `False`，避免下游分支对 None/False 语义理解不一致。
        if highlight is None:
            highlight = False

        # 先确定检索边界：搜哪些 KB、哪些文档、是否只搜可用文档等。
        filters = self.get_filters(req)
        
        # 只有“无问题直接列内容”场景才主要依赖字段排序。
        # 真正的问答检索排序，更多依赖全文分数、向量分数和重排分数。
        orderBy = OrderByExpr()

        # 底层存储通常吃 offset + limit，这里把页码统一折算掉。
        pg = int(req.get("page", 1)) - 1      # 页码（从0开始）
        topk = int(req.get("topk", 1024))     # 单次检索数量
        ps = int(req.get("size", topk))        # 每页大小
        offset, limit = pg * ps, ps            # 偏移量和限制

        # 这里列出来的字段，不只是“返回给前端看”的字段。
        # 它们同时还承担后续重排、引用补标、父子块合并、聚合统计等作用。
        src = req.get("fields",
                      ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", 
                       "important_kwd", "position_int", "doc_id", "chunk_order_int", 
                       "page_num_int", "top_int", "create_timestamp_flt", 
                       "knowledge_graph_kwd", "question_kwd", "question_tks", 
                       "doc_type_kwd", "available_int", "content_with_weight", 
                       "mom_id", PAGERANK_FLD, TAG_FLD, "row_id()"])
        
        # 用集合收集关键词，避免后续高亮阶段重复。
        kwds = set([])

        qst = req.get("question", "")  # 获取用户问题
        q_vec = []                     # 保存查询向量，后续重排阶段还要继续使用。
        
        # 情况 1：没有自然语言问题。
        # 这类调用更像后台浏览 / 调试接口，而不是标准 RAG 问答检索。
        if not qst:
            if req.get("sort"):
                # 这里尽量还原 chunk 在原文档中的自然顺序，便于阅读和调试。
                orderBy.asc("chunk_order_int")
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.get_total(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        
        # 情况 2：有自然语言问题，进入标准 RAG 检索路径。
        else:
            # 高亮默认落在正文和标题分词字段上，也支持调用方显式指定字段列表。
            highlightFields = ["content_ltks", "title_tks"]
            if not highlight:
                highlightFields = []
            elif isinstance(highlight, list):
                highlightFields = highlight
            
            # 先把自然语言问题转换成全文检索表达式，同时提取一批关键词。
            # 这一步由 `query.FulltextQueryer` 负责。
            matchText, keywords = self.qryr.question(qst, min_match=0.3)
            
            # 子情况 2.1：没有 embedding 模型，只能走纯全文检索。
            if emb_mdl is None:
                matchExprs = [matchText]
                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, 
                                            matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            
            # 子情况 2.2：全文 + 向量混合召回，这是最典型的 RAG 路径。
            else:
                # 把问题编码成查询向量。
                matchDense = await self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
                q_vec = matchDense.embedding_data
                
                # Infinity 在部分场景下会自己处理向量字段；
                # ES 路径下后续重排要手工读取文档侧向量，因此这里要补上对应列。
                if not settings.DOC_ENGINE_INFINITY:
                    src.append(f"q_{len(q_vec)}_vec")

                # 构建融合表达式：0.05 * 全文分数 + 0.95 * 向量分数
                # 当前融合策略固定为"少量全文分数 + 大量向量分数"
                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
                matchExprs = [matchText, matchDense, fusionExpr]

                # 执行混合检索
                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, 
                                            matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))

                # 第一次召回完全落空时，做一次保守的兜底重试。
                # 这一步的目的不是“保证一定准”，而是尽量避免问答链路直接没有候选材料可用。
                if total == 0:
                    if filters.get("doc_id"):
                        # 如果调用方已经把文档范围锁死了，就直接在该文档里取内容，
                        # 不再坚持全文 / 向量条件必须命中。
                        res = await thread_pool_exec(self.dataStore.search, src, [], filters, [], 
                                                    orderBy, offset, limit, idx_names, kb_ids)
                        total = self.dataStore.get_total(res)
                    else:
                        # 否则降低全文匹配门槛，并适度放宽向量阈值，再试一次。
                        matchText, _ = self.qryr.question(qst, min_match=0.1)
                        # 同时适度放宽向量相似度条件
                        matchDense.extra_options["similarity"] = 0.17
                        res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, 
                                                    [matchText, matchDense, fusionExpr],
                                                    orderBy, offset, limit, idx_names, kb_ids,
                                                    rank_feature=rank_feature)
                        total = self.dataStore.get_total(res)
                    logging.debug("Dealer.search 2 TOTAL: {}".format(total))

            # 收集关键词时把细粒度 token 也并进来，便于后续高亮和调试观察。
            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)

        logging.debug(f"TOTAL: {total}")
        
        # 把底层搜索结果统一抽取成 SearchResult，供后续重排和组装结果继续使用。
        ids = self.dataStore.get_doc_ids(res)
        keywords = list(kwds)
        highlight = self.dataStore.get_highlight(res, keywords, "content_with_weight")
        aggs = self.dataStore.get_aggregation(res, "docnm_kwd")
        
        return self.SearchResult(
            total=total,
            ids=ids,
            query_vector=q_vec,
            aggregation=aggs,
            highlight=highlight,
            field=self.dataStore.get_fields(res, src + ["_score"]),
            keywords=keywords
        )

    @staticmethod
    def trans2floats(txt):
        """将制表符分隔的字符串转换为浮点数组。
        
        Args:
            txt: 制表符分隔的字符串
            
        Returns:
            list: 浮点数组
        """
        # 底层存储里有些向量字段会被序列化成 `1.2\t3.4\t...` 这种字符串，
        # 这里统一把它们恢复成浮点数组，供相似度计算使用。
        return [get_float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9,
                         chunk_meta=None):
        """给大模型回答自动添加引用标注。
        
        通过计算回答中每个句子与检索到的chunk之间的混合相似度（向量+token），
        自动为回答中的句子添加引用来源标注。
        
        Args:
            answer: 大模型生成的回答文本
            chunks: 检索到的chunk文本列表
            chunk_v: chunk的向量表示列表
            embd_mdl: 嵌入模型实例
            tkweight: token相似度权重（默认0.1）
            vtweight: 向量相似度权重（默认0.9）
            chunk_meta: chunk元数据（可选）
            
        Returns:
            tuple: (添加引用后的回答, 引用的chunk索引集合)
            
        算法流程：
        1. 将回答按句子切分（保留代码块）
        2. 对每个句子编码为向量
        3. 计算每个句子与所有chunk的混合相似度
        4. 根据相似度阈值确定引用关系
        5. 生成带引用标注的回答
        """
        # 验证输入长度一致
        assert len(chunks) == len(chunk_v)
        
        # 如果没有chunk，直接返回原回答
        if not chunks:
            return answer, set([])
        
        # 步骤 1：先把回答切成“可比较的片段”。
        # 这里要特别保留代码块，避免把一段代码拆成很多句子，导致引用补标非常混乱。
        pieces = re.split(r"(```)", answer)
        
        if len(pieces) >= 3:
            # 处理包含代码块的情况
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    # 找到代码块起始标记
                    st = i
                    i += 1
                    # 找到代码块结束标记
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    # 整个代码块作为一个整体片段参与后续流程。
                    # 这样即使代码块很长，也不会在句子切分阶段被误拆。
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    # 普通文本按句子切分。
                    # 句子边界正则兼容阿拉伯语标点，避免多语言回答时切句失真。
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            # 直接按句子切分
            pieces = re.split(r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", answer)
        
        # 步骤 2：把被正则拆出来的标点重新并回前一句。
        # 否则后面做 embedding 时会出现大量只有标点或过短片段的噪声。
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！،؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        
        # 步骤 3：过滤过短片段。
        # 过短文本的 embedding 和 token 匹配都很不稳定，拿来补引用往往副作用更大。
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        
        logging.debug("{} => {}".format(answer, pieces_))
        
        if not pieces_:
            return answer, set([])

        # 步骤 4：把回答片段编码成向量，后面会逐句和候选 chunk 做比对。
        ans_v, _ = embd_mdl.encode(pieces_)
        
        # 步骤 5：对齐回答向量和 chunk 向量的维度。
        # 如果历史数据或不同模型写入过不一致的向量维度，这里至少要保证计算阶段不崩。
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0] * len(ans_v[0])
                logging.warning(
                    "The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        # 步骤 6：把 chunk 做分词，给 token 相似度计算做准备。
        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
                      for ck in chunks]
        
        # 步骤 7：逐句计算“回答片段 vs. 候选 chunk”的混合相似度。
        # 这里不会一上来用非常低的阈值，而是先从较高阈值开始，逐步下调。
        # 这样可以优先拿到最稳妥的引用，避免一开始就把很多边缘 chunk 也标进去。
        cites = {}           # 存储引用关系 {句子索引: [chunk索引列表]}
        thr = 0.63           # 初始相似度阈值
        
        # 动态调阈值的策略是：
        # - 只要还一个引用都没找到，就继续降低阈值；
        # - 一旦已经找到过引用，就停止继续放宽，避免引用面无限扩大。
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                # 计算混合相似度（向量相似度 + token相似度）
                sim, tksim, vtsim = self.qryr.hybrid_similarity(
                    ans_v[i],                           # 句子向量
                    chunk_v,                            # chunk向量列表
                    rag_tokenizer.tokenize(self.qryr.rmWWW(pieces_[i])).split(),  # 句子分词
                    chunks_tks,                          # chunk分词列表
                    tkweight, vtweight                   # 权重配置
                )
                
                # 这里不用固定阈值截断，而是取该句“最佳候选分数”的 99% 作为相对门槛。
                # 这么做的效果是：每句话优先绑定它自己最接近的那几个 chunk。
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                
                if mx < thr:
                    continue
                
                # 最多记录 4 个引用，避免回答里一段话挂太多来源，影响可读性。
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                # 如果拿得到 chunk 元数据，就优先生成“文档名 / 页码”这种可读引用；
                # 否则退化成内部 ID。
                if chunk_meta and int(c) < len(chunk_meta):
                    meta = chunk_meta[int(c)]
                    doc_name = meta.get("docnm_kwd", "")
                    positions = meta.get("top_int") or meta.get("positions") or []
                    page = positions[0] if positions else None
                    if doc_name and page is not None:
                        res += f" [来源：{doc_name}，第 {page} 页]"
                    elif doc_name:
                        res += f" [来源：{doc_name}]"
                    else:
                        res += f" [ID:{c}]"
                else:
                    res += f" [ID:{c}]"
                seted.add(c)

        return res, seted

    def _rank_feature_scores(self, query_rfea, search_res):
        """计算标签类 rank feature 的附加得分。

        该方法实现了基于标签匹配的附加评分机制：
        - 将查询的标签权重与文档的标签权重进行向量点积
        - 进行归一化处理（除以两个向量的模长）
        - 最终得分乘以10作为放大系数

        Args:
            query_rfea (dict): 查询的标签权重字典 {tag: score}
            search_res (SearchResult): 检索结果对象

        Returns:
            np.array: 每个文档的标签匹配得分数组
        """
        # 这个附加分并不是主召回分数，而是“标签匹配奖励项”。
        # 适合那种已经给问题和文档都打过标签的场景。
        rank_fea = []

        # 如果没有查询标签特征，返回全0数组
        if not query_rfea:
            return np.array([0 for _ in range(len(search_res.ids))])

        # 计算查询标签向量的模长（归一化分母）
        q_denor = np.sqrt(np.sum([s * s for t, s in query_rfea.items()]))

        # 遍历每个检索结果，计算标签匹配得分
        for i in search_res.ids:
            nor, denor = 0, 0
            # 如果文档没有标签字段，得分设为0
            if not search_res.field[i].get(TAG_FLD):
                rank_fea.append(0)
                continue
            # 文档侧标签是字符串存下来的字典，这里先解析出来再算点积。
            for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
                # 如果标签在查询中存在，累加点积
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                # 累加文档标签权重的平方和
                denor += sc * sc
            # 计算归一化得分
            if denor == 0:
                rank_fea.append(0)
            else:
                rank_fea.append(nor / np.sqrt(denor) / q_denor)

        # 将得分放大10倍后返回
        return np.array(rank_fea) * 10.

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: dict | None = None
               ):
        """使用混合相似度对候选 chunk 做二次排序（规则排序）。

        该方法实现基于规则的重排序，计算每个chunk与查询的混合相似度：
        混合相似度 = token相似度 * tkweight + 向量相似度 * vtweight

        权重设计原则：
        - 默认 tkweight=0.3：token相似度占30%
        - 默认 vtweight=0.7：向量相似度占70%

        Args:
            sres (SearchResult): 初步检索结果
            query (str): 用户查询
            tkweight (float): token相似度权重（默认0.3）
            vtweight (float): 向量相似度权重（默认0.7）
            cfield (str): 内容字段名（默认"content_ltks"）
            rank_feature (dict): 标签特征权重（可选）

        Returns:
            tuple: (混合相似度数组, token相似度数组, 向量相似度数组)
        """
        # 先从问题里重新拿一组关键词。
        # 注意：这里的关键词不是拿来做底层召回，而是服务于规则重排时的 token 相似度计算。
        _, keywords = self.qryr.question(query)
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        ins_embd = []
        for chunk_id in sres.ids:
            # 候选结果里如果没有对应向量字段，就退化成零向量；
            # 这样至少不至于在后续相似度阶段直接报错。
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [get_float(v) for v in vector.split("\t")]
            ins_embd.append(vector)
        if not ins_embd:
            return [], [], []

        # 统一把 `important_kwd` 调整成 list，避免下游拼接时出现字符串被逐字符展开的问题。
        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            # 规则重排的 token 特征不是简单拿正文分词，而是给不同来源的 token 不同权重：
            # - 正文词：基础权重
            # - 标题词：加倍
            # - important_kwd：更高权重
            # - question_tks：最高一档，强调问答型 chunk 的问题面
            content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        ## 计算标签类 rank feature 的附加得分。
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.4,
                        vtweight=0.6, cfield="content_ltks",
                        rank_feature: dict | None = None):
        """使用重排序模型对候选 chunk 做二次排序。

        最终分数计算公式：
        最终分数 = vtweight * reranker语义分 + tkweight * token相似度 + rank_feature附加分

        默认权重配置（语义与词面并重）：
        - vtweight=0.6：模型语义相似度占60%
        - tkweight=0.4：token相似度占40%

        Args:
            rerank_mdl: 重排序模型实例
            sres (SearchResult): 初步检索结果
            query (str): 用户查询
            tkweight (float): token相似度权重（默认0.4）
            vtweight (float): 模型语义相似度权重（默认0.6）
            cfield (str): 内容字段名（默认"content_ltks"）
            rank_feature (dict): 标签特征权重（可选）

        Returns:
            tuple: (综合分数数组, token相似度数组, 模型语义相似度数组)
        """
        # 和规则重排类似，这里也需要先准备一份查询关键词，
        # 因为最终分数依然保留了 token 相似度这一项。
        _, keywords = self.qryr.question(query)

        # 统一把重要关键词转成 list，避免后续拼接文档内容时出现类型不一致。
        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            # 模型重排路径下，这里的 token 特征比规则重排更克制一些：
            # 只保留正文、标题和重要关键词，不再额外给 question_tks 很高权重。
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        # token 相似度负责补足词面精确匹配，
        # reranker 语义分负责表达更强的句子级语义相关性。
        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim, _ = rerank_mdl.similarity(query, [remove_redundant_spaces(" ".join(tks)) for tks in ins_tw])

        # 标签特征加成是额外增益项，不参与两项主分数的线性组合。
        rank_fea = self._rank_feature_scores(rank_feature, sres) if rank_feature else np.zeros(len(sres.ids))

        composite = vtweight * np.array(vtsim) + tkweight * np.array(tksim)
        return composite + rank_fea, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        """对 `qryr.hybrid_similarity` 的薄封装。

        这个方法主要是为了让上层调用时不用手动分词，
        传入原始文本后由这里统一转成 token 列表。
        """
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           rag_tokenizer.tokenize(ans).split(),
                                           rag_tokenizer.tokenize(inst).split())

    async def retrieval(
            self,
            question,
            embd_mdl,
            kb_ids,
            page,
            page_size,
            similarity_threshold=0.2,
            vector_similarity_weight=0.3,
            top=1024,
            doc_ids=None,
            aggs=True,
            rerank_mdl=None,
            highlight=False,
            rank_feature: dict | None = None,
    ):
        """对外暴露的顶层检索接口，提供完整的检索服务。

        这是 RAG 系统对外暴露的核心检索接口，封装了完整的检索流程：
        1. 构建检索请求
        2. 执行底层混合检索（全文 + 向量）
        3. 使用重排序模型或规则进行二次排序
        4. 分页处理并返回结果
        5. 可选的文档聚合统计

        Args:
            question (str): 已经经过多轮归并、翻译、关键词增强后的最终检索问题
            embd_mdl: 嵌入模型实例（用于向量检索）
            kb_ids (list): 知识库ID列表，知识库范围
            page (int): 页码（从1开始）
            page_size (int): 每页大小
            similarity_threshold (float): 相似度阈值（默认0.2）
            vector_similarity_weight (float): 向量相似度在最终分数里的占比（默认0.3）
            top (int): 初步检索返回数量（默认1024）
            doc_ids (list): 如果限定了文档范围，就只在这些文档里检索（可选）
            aggs (bool): 是否返回文档聚合统计（默认True）
            rerank_mdl: 重排序模型实例（可选）
            highlight (bool): 是否返回高亮结果（默认False）
            rank_feature (dict): 标签特征权重（可选）

        Returns:
            dict: 检索结果字典
                - total: 总结果数
                - chunks: chunk列表，包含相似度分数和元数据
                - doc_aggs: 文档聚合统计列表
        """
        # `ranks` 是这个顶层接口最终返回给上层编排逻辑的统一结构。
        # 上层一般只关心三件事：
        # 1. 一共召回了多少条；
        # 2. 当前页可用的 chunk 列表；
        # 3. 命中了哪些文档及各自命中次数。
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        # ========== 输入校验 ==========
        if not question:
            return ranks

        # ========== 计算重排序候选集大小 ==========
        # 让候选集大小覆盖整页，避免重排后分页结果不稳定
        # 公式：ceil(64 / page_size) * page_size，最小30
        RERANK_LIMIT = math.ceil(64 / page_size) * page_size if page_size > 1 else 1
        RERANK_LIMIT = max(30, RERANK_LIMIT)

        # 这里先向底层拿一个“更宽的候选集”，而不是直接拿最终页大小。
        # 原因是：底层召回顺序和最终重排顺序不完全一样，如果候选集太小，重排空间就不够。
        # 构建请求
        req = {
            "kb_ids": kb_ids,           # 知识库ID列表
            "doc_ids": doc_ids,         # 限定文档ID（可选）
            "page": math.ceil(page_size * page / RERANK_LIMIT),  # 计算实际查询页码
            "size": RERANK_LIMIT,       # 候选集大小
            "question": question,       # 用户问题
            "vector": True,             # 启用向量检索
            "topk": top,                # 初步检索数量
            "similarity": similarity_threshold,  # 相似度阈值
            "available_int": 1,         # 只检索可用文档
        }

        # ========== 执行底层混合检索 ==========
        sres = await self.search(req, [index_name()], kb_ids, embd_mdl, highlight,
                           rank_feature=rank_feature)

        # ========== 二次排序（重排序） ==========
        if rerank_mdl and sres.total > 0:
            # 如果配置了 reranker，就优先用模型重排。
            # 这一步通常比底层融合分数更贴近最终问答效果。
            sim, tsim, vsim = self.rerank_by_model(
                rerank_mdl,
                sres,
                question,
                1 - vector_similarity_weight,  # token权重 = 1 - 向量权重
                vector_similarity_weight,       # 向量权重
                rank_feature=rank_feature,
            )
        else:
            # 否则退化成规则重排。
            if settings.DOC_ENGINE_INFINITY:
                # Infinity 路径下，底层返回的 `_score` 已经可以直接作为融合分数使用。
                sim = [sres.field[id].get("_score", 0.0) for id in sres.ids]
                sim = [s if s is not None else 0.0 for s in sim]
                tsim = sim
                vsim = sim
            else:
                # ES 路径下，文档侧向量和 token 特征要在这里手工再算一次综合分。
                sim, tsim, vsim = self.rerank(
                    sres,
                    question,
                    1 - vector_similarity_weight,
                    vector_similarity_weight,
                    rank_feature=rank_feature,
                )

        sim_np = np.array(sim, dtype=np.float64)
        if sim_np.size == 0:
            ranks["doc_aggs"] = []
            return ranks

        # ========== 重排后再做一次阈值过滤和分页 ==========
        # `argsort(* -1)` 表示按分数从高到低排序。
        sorted_idx = np.argsort(sim_np * -1)

        # 如果完全关闭了向量权重，就不再强行要求分数超过“向量阈值”。
        post_threshold = 0.0 if vector_similarity_weight <= 0 else similarity_threshold

        # 如果调用方明确限定了 doc_ids，一般是在“我就想看这些文档里有什么”的语境下，
        # 这时再按阈值裁掉结果，反而可能把本来就希望看到的内容过滤掉。
        if doc_ids:
            post_threshold = 0.0

        valid_idx = [int(i) for i in sorted_idx if sim_np[i] >= post_threshold]
        filtered_count = len(valid_idx)
        ranks["total"] = int(filtered_count)

        if filtered_count == 0:
            ranks["doc_aggs"] = []
            return ranks

        # 这里的分页要基于“重排后的候选集”来切，而不是直接沿用底层搜索页码。
        # 否则第一页和第二页可能分别在不同候选窗口里重排，导致整体顺序不稳定。
        max_pages = max(RERANK_LIMIT // max(page_size, 1), 1)
        page_index = (page - 1) % max_pages
        begin = page_index * page_size
        end = begin + page_size
        page_idx = valid_idx[begin:end]

        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim

        for i in page_idx:
            # 这一段是在把 SearchResult 里的原始字段重新整理成上层更容易消费的 chunk 结构。
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")

            position_int = chunk.get("position_int", [])
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "tag_kwd": chunk.get("tag_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": float(sim_np[i]),
                "vector_similarity": float(vsim[i]),
                "term_similarity": float(tsim[i]),
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),
                "mom_id": chunk.get("mom_id", ""),
                "row_id": chunk.get("row_id()"),
            }
            if highlight and sres.highlight:
                # 如果底层能返回高亮内容，就优先用高亮版本；
                # 否则回退到原始正文，保证字段总是可用。
                if id in sres.highlight:
                    d["highlight"] = remove_redundant_spaces(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
            ranks["chunks"].append(d)

        if aggs:
            # 文档聚合统计用于前端展示“哪些文档命中更多”，
            # 也方便上层后续做引用范围收缩。
            for i in valid_idx:
                id = sres.ids[i]
                chunk = sres.field[id]
                dnm = chunk.get("docnm_kwd", "")
                did = chunk.get("doc_id", "")
                if dnm not in ranks["doc_aggs"]:
                    ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
                ranks["doc_aggs"][dnm]["count"] += 1

            ranks["doc_aggs"] = [
                {
                    "doc_name": k,
                    "doc_id": v["doc_id"],
                    "count": v["count"],
                }
                for k, v in sorted(
                    ranks["doc_aggs"].items(),
                    key=lambda x: x[1]["count"] * -1,
                )
            ]
        else:
            ranks["doc_aggs"] = []

        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        """执行 SQL 检索。

        这条路径主要服务于结构化字段映射场景。
        它不是标准向量 RAG 主路，但会在某些知识库已建立字段映射时被优先尝试。
        """
        tbl = self.dataStore.sql(sql, fetch_size, format)
        return tbl

    def chunk_list(self, doc_id: str,
                   kb_ids: list[str], max_count=1024,
                   offset=0,
                   fields=["docnm_kwd", "content_with_weight", "img_id"],
                   sort_by_position: bool = False):
        """按文档列出 chunk。

        这个接口更偏工具型能力，常用于：
        - 调试某篇文档实际切成了哪些块；
        - 根据命中的 doc_id 把整篇文档的 chunk 拉出来复查；
        - TOC / parent-child 之类增强逻辑里按文档回读内容。
        """
        condition = {"doc_id": doc_id}

        # 如果要按文档位置排序，就补齐位置相关字段，避免后面拿不到排序键。
        fields_set = set(fields or [])
        if sort_by_position:
            for need in ("page_num_int", "position_int", "top_int"):
                if need not in fields_set:
                    fields_set.add(need)
        fields = list(fields_set)

        orderBy = OrderByExpr()
        if sort_by_position:
            orderBy.asc("page_num_int")
            orderBy.asc("position_int")
            orderBy.asc("top_int")

        # 这里用批次循环而不是一次性全取，是为了避免长文档 chunk 太多时单次查询过大。
        res = []
        bs = 128
        for p in range(offset, max_count, bs):
            limit = min(bs, max_count - p)
            if limit <= 0:
                break
            es_res = self.dataStore.search(fields, [], condition, [], orderBy, p, limit, index_name(),
                                           kb_ids)
            dict_chunks = self.dataStore.get_fields(es_res, fields)
            for id, doc in dict_chunks.items():
                doc["id"] = id
            if dict_chunks:
                res.extend(dict_chunks.values())
            chunk_count = len(dict_chunks)
            if chunk_count == 0 or chunk_count < limit:
                break
        return res

    def all_tags(self, kb_ids: list[str], S=1000):
        """统计知识库中出现过的所有标签及其频次。"""
        if not self.dataStore.index_exist(index_name(), kb_ids[0]):
            return []
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(), kb_ids, ["tag_kwd"])
        return self.dataStore.get_aggregation(res, "tag_kwd")

    def all_tags_in_portion(self, kb_ids: list[str], S=1000):
        """把标签频次归一化成平滑后的先验分布。"""
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(), kb_ids, ["tag_kwd"])
        res = self.dataStore.get_aggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        # 加 1 / 加 S 属于平滑处理，避免低频或未见标签把分母直接打崩。
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        """根据相似内容为文档 chunk 生成标签特征。"""
        idx_nm = index_name()
        # 这里用“标题 + 正文分词 + important_kwd”构造查询，
        # 再看历史数据里哪些标签最常与这类内容一起出现。
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []),
                                        keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        # 这里本质上是在算“标签在相关内容里的相对提升度”，
        # 再结合全局先验频率，尽量压制那种到处都出现的泛化标签。
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a.replace(".", "_"): c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        """为查询问题生成标签特征。"""
        idx_nms = index_name()
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        # 这里返回的是查询侧标签权重，后续会在 `_rank_feature_scores` 里和文档标签做匹配加分。
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        return {a.replace(".", "_"): max(1, c) for a, c in tag_fea}

    async def retrieval_by_toc(self, query: str, chunks: list[dict], chat_mdl, topn: int = 6):
        """结合目录结构再次补召回，适合长文档。"""
        # 延迟导入是为了避免 `search.py` <-> `generator.py` 的循环依赖。
        from rag.prompts.generator import relevant_chunks_with_toc
        if not chunks:
            return []
        idx_nms = index_name()
        ranks, doc_id2kb_id = {}, {}
        for ck in chunks:
            # 先找出“当前召回里最有代表性的一篇文档”。
            # 这里用 chunk 相似度之和做一个粗略打分，认为它最可能包含结构化 TOC 信息。
            if ck["doc_id"] not in ranks:
                ranks[ck["doc_id"]] = 0
            ranks[ck["doc_id"]] += ck["similarity"]
            doc_id2kb_id[ck["doc_id"]] = ck["kb_id"]
        doc_id = sorted(ranks.items(), key=lambda x: x[1] * -1.)[0][0]
        kb_ids = [doc_id2kb_id[doc_id]]
        es_res = self.dataStore.search(["content_with_weight"], [], {"doc_id": doc_id, "toc_kwd": "toc"}, [],
                                       OrderByExpr(), 0, 128, idx_nms,
                                       kb_ids)
        toc = []
        dict_chunks = self.dataStore.get_fields(es_res, ["content_with_weight"])
        for _, doc in dict_chunks.items():
            try:
                # TOC chunk 里一般是 JSON 形式的目录结构，这里把它展开成统一列表。
                toc.extend(json.loads(doc["content_with_weight"]))
            except Exception as e:
                logging.exception(e)
        if not toc:
            return chunks

        # 让 LLM 基于目录理解“哪些章节最可能回答当前问题”，再反向补召回对应 chunk。
        ids = await relevant_chunks_with_toc(query, toc, chat_mdl, topn * 2)
        if not ids:
            return chunks

        vector_size = 1024
        id2idx = {ck["chunk_id"]: i for i, ck in enumerate(chunks)}
        for cid, sim in ids:
            if cid in id2idx:
                # 如果该 chunk 原本就命中了，就把 TOC 相关性作为额外加分。
                chunks[id2idx[cid]]["similarity"] += sim
                continue
            chunk = self.dataStore.get(cid, idx_nms[0], kb_ids)
            if not chunk:
                continue
            d = {
                "chunk_id": cid,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": doc_id,
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim,
                "vector_similarity": sim,
                "term_similarity": sim,
                "vector": [0.0] * vector_size,
                "positions": chunk.get("position_int", []),
                "doc_type_kwd": chunk.get("doc_type_kwd", "")
            }
            for k in chunk.keys():
                if k[-4:] == "_vec":
                    d["vector"] = chunk[k]
                    vector_size = len(chunk[k])
                    break
            chunks.append(d)

        # TOC 补召回完再整体重排一次，只保留 topn。
        return sorted(chunks, key=lambda x: x["similarity"] * -1)[:topn]

    def retrieval_by_children(self, chunks: list[dict]):
        """把命中的子 chunk 合并回父 chunk，提供更完整上下文。"""
        if not chunks:
            return []
        idx_nms = index_name()
        mom_chunks = defaultdict(list)
        i = 0
        while i < len(chunks):
            ck = chunks[i]
            mom_id = ck.get("mom_id")
            if not isinstance(mom_id, str) or not mom_id.strip():
                i += 1
                continue
            # 这里把命中的子块先临时摘出来，按父块 ID 分组。
            mom_chunks[ck["mom_id"]].append(chunks.pop(i))

        if not mom_chunks:
            return chunks

        if not chunks:
            chunks = []

        vector_size = 1024
        for id, cks in mom_chunks.items():
            # 再把父块本身从存储层拉回来，用它的完整正文替换掉子块碎片。
            # 这样模型最后看到的是更完整的上下文，而不是很多割裂的小片段。
            chunk = self.dataStore.get(id, idx_nms[0], [ck["kb_id"] for ck in cks])
            d = {
                "chunk_id": id,
                "content_ltks": " ".join([ck["content_ltks"] for ck in cks]),
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": chunk["doc_id"],
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "kb_id": chunk["kb_id"],
                "important_kwd": [kwd for ck in cks for kwd in ck.get("important_kwd", [])],
                "image_id": chunk.get("img_id", ""),
                "similarity": np.mean([ck["similarity"] for ck in cks]),
                "vector_similarity": np.mean([ck["similarity"] for ck in cks]),
                "term_similarity": np.mean([ck["similarity"] for ck in cks]),
                "vector": [0.0] * vector_size,
                "positions": chunk.get("position_int", []),
                "doc_type_kwd": chunk.get("doc_type_kwd", "")
            }
            for k in cks[0].keys():
                if k[-4:] == "_vec":
                    # 父块这里沿用子块已有的向量维度信息，避免后续字段缺失。
                    d["vector"] = cks[0][k]
                    vector_size = len(cks[0][k])
                    break
            chunks.append(d)

        # 回收完父块后按相似度重新排序。
        return sorted(chunks, key=lambda x: x["similarity"] * -1)
