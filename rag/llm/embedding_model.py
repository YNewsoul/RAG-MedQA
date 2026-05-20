"""向量嵌入模型适配层。

这里把不同厂商/部署方式的向量模型封装成统一 `encode / encode_queries` 接口，
方便上层检索逻辑直接调用。

支持的嵌入模型：
- OpenAI：text-embedding-ada-002
- Azure OpenAI
- 百度文心一言
- 阿里云通义千问
- 百川智能
- LocalAI（本地部署）
- TEI（文本嵌入接口服务）
- 智谱AI
- Ollama（本地部署）
- HuggingFace（本地部署）
"""

import json
import os
import threading
from abc import ABC
from urllib.parse import urljoin

import dashscope
import numpy as np
import requests
from ollama import Client
from openai import OpenAI
from zhipuai import ZhipuAI

from common.log_utils import log_exception
from common.token_utils import num_tokens_from_string, truncate, total_token_count_from_response
from common import settings
import logging
import base64


class Base(ABC):
    """向量嵌入模型抽象基类。
    
    定义所有嵌入模型必须实现的接口，确保上层代码可以统一调用。
    """
    def __init__(self, key, model_name, **kwargs):
        """抽象基类构造函数，仅用于保持接口一致。"""
        pass

    def encode(self, texts: list):
        """将文本列表编码为向量。
        
        Args:
            texts: 文本列表
            
        Returns:
            tuple: (向量数组, token计数)
        """
        raise NotImplementedError("Please implement encode method!")

    def encode_queries(self, text: str):
        """将单条查询文本编码为向量。
        
        Args:
            text: 查询文本
            
        Returns:
            tuple: (向量数组, token计数)
        """
        raise NotImplementedError("Please implement encode method!")


class BuiltinEmbed(Base):
    """内置嵌入模型封装（基于TEI服务）。
    
    通过TEI（Text Embedding Inference）服务提供向量嵌入能力，
    支持多种开源模型如Qwen3-Embedding、BGE-M3等。
    """
    _FACTORY_NAME = "Builtin"
    MAX_TOKENS = {"Qwen/Qwen3-Embedding-0.6B": 30000, "BAAI/bge-m3": 8000, "BAAI/bge-small-en-v1.5": 500}
    _model = None      # 单例模型实例
    _model_name = ""   # 模型名称
    _max_tokens = 500  # 最大token数
    _model_lock = threading.Lock()  # 线程锁（保证单例线程安全）

    def __init__(self, key, model_name, **kwargs):
        """初始化内置嵌入模型。"""
        logging.info(f"Initialize BuiltinEmbed according to settings.EMBEDDING_CFG: {settings.EMBEDDING_CFG}")
        embedding_cfg = settings.EMBEDDING_CFG
        # 懒加载：仅当使用TEI配置时才创建模型
        if not BuiltinEmbed._model and "tei-" in os.getenv("COMPOSE_PROFILES", ""):
            with BuiltinEmbed._model_lock:
                BuiltinEmbed._model_name = settings.EMBEDDING_MDL
                BuiltinEmbed._max_tokens = BuiltinEmbed.MAX_TOKENS.get(settings.EMBEDDING_MDL, 500)
                BuiltinEmbed._model = HuggingFaceEmbed(embedding_cfg["api_key"], settings.EMBEDDING_MDL, base_url=embedding_cfg["base_url"])
        self._model = BuiltinEmbed._model
        self._model_name = BuiltinEmbed._model_name
        self._max_tokens = BuiltinEmbed._max_tokens

    def encode(self, texts: list):
        """批量编码文本为向量。"""
        batch_size = 16
        # TEI 服务端会自动裁剪超长输入，因此这里只做批处理
        token_count = 0
        ress = None
        for i in range(0, len(texts), batch_size):
            embeddings, token_count_delta = self._model.encode(texts[i : i + batch_size])
            token_count += token_count_delta
            if ress is None:
                ress = embeddings
            else:
                ress = np.concatenate((ress, embeddings), axis=0)
        return ress, token_count

    def encode_queries(self, text: str):
        """编码单条查询文本。"""
        return self._model.encode_queries(text)


class OpenAIEmbed(Base):
    """OpenAI 嵌入模型封装。
    
    使用 OpenAI API 提供向量嵌入服务，支持自定义 base_url（兼容代理）。
    """
    _FACTORY_NAME = "OpenAI"

    def __init__(self, key, model_name="text-embedding-ada-002", base_url="https://api.openai.com/v1"):
        """初始化 OpenAI 嵌入客户端。
        
        Args:
            key: API密钥
            model_name: 模型名称（默认text-embedding-ada-002）
            base_url: API基础URL
        """
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def encode(self, texts: list):
        """批量编码文本为向量。
        
        OpenAI embedding 接口单批次不宜过大，这里保守控制在16条。
        """
        batch_size = 16
        texts = [truncate(t, 8191) for t in texts]  # 截断到最大长度
        ress = []
        total_tokens = 0
        for i in range(0, len(texts), batch_size):
            res = self.client.embeddings.create(input=texts[i : i + batch_size], 
                                               model=self.model_name, 
                                               encoding_format="float", 
                                               extra_body={"drop_params": True})
            try:
                ress.extend([d.embedding for d in res.data])
                total_tokens += total_token_count_from_response(res)
            except Exception as _e:
                log_exception(_e, res)
                raise Exception(f"Error: {res}")
        return np.array(ress), total_tokens

    def encode_queries(self, text):
        """编码单条查询文本。"""
        res = self.client.embeddings.create(input=[truncate(text, 8191)], 
                                           model=self.model_name, 
                                           encoding_format="float", 
                                           extra_body={"drop_params": True})
        try:
            return np.array(res.data[0].embedding), total_token_count_from_response(res)
        except Exception as _e:
            log_exception(_e, res)
            raise Exception(f"Error: {res}")


class LocalAIEmbed(Base):
    """LocalAI 嵌入模型封装。
    
    支持 LmStudio 等本地部署的嵌入模型，使用 OpenAI 兼容接口。
    """
    _FACTORY_NAME = "LocalAI"

    def __init__(self, key, model_name, base_url):
        """初始化 LocalAI 嵌入客户端。
        
        Args:
            key: API密钥（本地模型通常不需要）
            model_name: 模型名称
            base_url: 本地服务URL
        """
        if not base_url:
            raise ValueError("Local embedding model url cannot be None")
        base_url = urljoin(base_url, "v1")
        self.client = OpenAI(api_key="empty", base_url=base_url)
        self.model_name = model_name.split("___")[0]

    def encode(self, texts: list):
        """批量编码文本为向量。
        
        LmStudio 本地 embedding 通常不返回可靠 token 用量，使用固定占位值1024。
        """
        batch_size = 16
        ress = []
        for i in range(0, len(texts), batch_size):
            res = self.client.embeddings.create(input=texts[i : i + batch_size], model=self.model_name)
            try:
                ress.extend([d.embedding for d in res.data])
            except Exception as _e:
                log_exception(_e, res)
                raise Exception(f"Error: {res}")
        # LmStudio 本地 embedding 通常不返回可靠 token 用量，这里使用固定占位值
        return np.array(ress), 1024

    def encode_queries(self, text):
        """编码单条查询文本。"""
        embds, cnt = self.encode([text])
        return np.array(embds[0]), cnt


class AzureEmbed(OpenAIEmbed):
    """Azure OpenAI 嵌入模型封装。
    
    继承自 OpenAIEmbed，适配 Azure OpenAI 服务。
    """
    _FACTORY_NAME = "Azure-OpenAI"

    def __init__(self, key, model_name, **kwargs):
        """初始化 Azure OpenAI 嵌入客户端。
        
        Args:
            key: JSON格式的密钥（包含api_key和api_version）
            model_name: 模型名称
            kwargs: 包含base_url等参数
        """
        from openai.lib.azure import AzureOpenAI

        api_key = json.loads(key).get("api_key", "")
        api_version = json.loads(key).get("api_version", "2024-02-01")
        self.client = AzureOpenAI(api_key=api_key, azure_endpoint=kwargs["base_url"], api_version=api_version)
        self.model_name = model_name


class BaiChuanEmbed(OpenAIEmbed):
    """百川智能嵌入模型封装。
    
    继承自 OpenAIEmbed，使用 OpenAI 兼容接口调用百川嵌入服务。
    """
    _FACTORY_NAME = "BaiChuan"

    def __init__(self, key, model_name="Baichuan-Text-Embedding", base_url="https://api.baichuan-ai.com/v1"):
        """初始化百川嵌入客户端。
        
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if not base_url:
            base_url = "https://api.baichuan-ai.com/v1"
        super().__init__(key, model_name, base_url)


class QWenEmbed(Base):
    """阿里云通义千问嵌入模型封装。
    
    使用阿里云 DashScope SDK 调用通义千问嵌入服务。
    """
    _FACTORY_NAME = "Tongyi-Qianwen"

    def __init__(self, key, model_name="text_embedding_v2", **kwargs):
        """初始化通义千问嵌入客户端。
        
        Args:
            key: API密钥
            model_name: 模型名称（默认text_embedding_v2）
        """
        self.key = key
        self.model_name = model_name

    def encode(self, texts: list):
        """批量编码文本为向量。
        
        支持重试机制，最多重试5次，每次间隔10秒。
        """
        import time
        import dashscope

        batch_size = 4  # 千问嵌入接口批次较小
        res = []
        token_count = 0
        texts = [truncate(t, 2048) for t in texts]
        for i in range(0, len(texts), batch_size):
            retry_max = 5
            resp = dashscope.TextEmbedding.call(model=self.model_name, 
                                               input=texts[i : i + batch_size], 
                                               api_key=self.key, 
                                               text_type="document")
            # 重试机制
            while (resp["output"] is None or resp["output"].get("embeddings") is None) and retry_max > 0:
                time.sleep(10)
                resp = dashscope.TextEmbedding.call(model=self.model_name, 
                                                   input=texts[i : i + batch_size], 
                                                   api_key=self.key, 
                                                   text_type="document")
                retry_max -= 1
            # 重试失败处理
            if retry_max == 0 and (resp["output"] is None or resp["output"].get("embeddings") is None):
                if resp.get("message"):
                    log_exception(ValueError(f"Retry_max reached, calling embedding model failed: {resp['message']}"))
                else:
                    log_exception(ValueError("Retry_max reached, calling embedding model failed"))
                raise
            try:
                embds = [[] for _ in range(len(resp["output"]["embeddings"]))]
                for e in resp["output"]["embeddings"]:
                    embds[e["text_index"]] = e["embedding"]
                res.extend(embds)
                token_count += total_token_count_from_response(resp)
            except Exception as _e:
                log_exception(_e, resp)
                raise
        return np.array(res), token_count

    def encode_queries(self, text):
        """编码单条查询文本（使用query类型）。"""
        resp = dashscope.TextEmbedding.call(model=self.model_name, input=text[:2048], api_key=self.key, text_type="query")
        try:
            return np.array(resp["output"]["embeddings"][0]["embedding"]), total_token_count_from_response(resp)
        except Exception as _e:
            log_exception(_e, resp)
            raise Exception(f"Error: {resp}")


class ZhipuEmbed(Base):
    """智谱AI嵌入模型封装。
    
    使用智谱AI官方SDK调用嵌入服务，支持embedding-2和embedding-3模型。
    """
    _FACTORY_NAME = "ZHIPU-AI"

    def __init__(self, key, model_name="embedding-2", **kwargs):
        """初始化智谱AI嵌入客户端。
        
        Args:
            key: API密钥
            model_name: 模型名称（embedding-2或embedding-3）
        """
        self.client = ZhipuAI(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list):
        """批量编码文本为向量。
        
        根据模型类型设置不同的最大长度：
        - embedding-2: 512 tokens
        - embedding-3: 3072 tokens
        """
        arr = []
        tks_num = 0
        MAX_LEN = -1
        if self.model_name.lower() == "embedding-2":
            MAX_LEN = 512
        if self.model_name.lower() == "embedding-3":
            MAX_LEN = 3072
        if MAX_LEN > 0:
            texts = [truncate(t, MAX_LEN) for t in texts]

        for txt in texts:
            res = self.client.embeddings.create(input=txt, model=self.model_name)
            try:
                arr.append(res.data[0].embedding)
                tks_num += total_token_count_from_response(res)
            except Exception as _e:
                log_exception(_e, res)
                raise Exception(f"Error: {res}")
        return np.array(arr), tks_num

    def encode_queries(self, text):
        """编码单条查询文本。"""
        res = self.client.embeddings.create(input=text, model=self.model_name)
        try:
            return np.array(res.data[0].embedding), total_token_count_from_response(res)
        except Exception as _e:
            log_exception(_e, res)
            raise Exception(f"Error: {res}")


class OllamaEmbed(Base):
    """Ollama 本地嵌入模型封装。
    
    使用 Ollama 客户端调用本地部署的嵌入模型，支持 API 密钥认证。
    """
    _FACTORY_NAME = "Ollama"

    _special_tokens = ["<|endoftext|>"]  # 需要移除的特殊token

    def __init__(self, key, model_name, **kwargs):
        """初始化 Ollama 嵌入客户端。
        
        Args:
            key: API密钥（可选）
            model_name: 模型名称
            kwargs: 包含base_url和ollama_keep_alive等参数
        """
        self.client = Client(host=kwargs["base_url"]) if not key or key == "x" else Client(host=kwargs["base_url"], headers={"Authorization": f"Bearer {key}"})
        self.model_name = model_name
        self.keep_alive = kwargs.get("ollama_keep_alive", int(os.environ.get("OLLAMA_KEEP_ALIVE", -1)))

    def encode(self, texts: list):
        """批量编码文本为向量。
        
        先移除 Ollama 侧可能误处理的特殊 token，再发起请求。
        Ollama 本地模型不返回可靠 token 计数，使用固定值128。
        """
        arr = []
        tks_num = 0
        for txt in texts:
            # 先移除 Ollama 侧可能误处理的特殊 token，再发起请求
            for token in OllamaEmbed._special_tokens:
                txt = txt.replace(token, "")
            res = self.client.embeddings(prompt=txt, model=self.model_name, options={"use_mmap": True}, keep_alive=self.keep_alive)
            try:
                arr.append(res["embedding"])
            except Exception as _e:
                log_exception(_e, res)
                raise Exception(f"Error: {res}")
            tks_num += 128
        return np.array(arr), tks_num

    def encode_queries(self, text):
        """编码单条查询文本。
        
        查询向量也做同样的清洗，保持和文档向量处理方式一致。
        """
        for token in OllamaEmbed._special_tokens:
            text = text.replace(token, "")
        res = self.client.embeddings(prompt=text, model=self.model_name, options={"use_mmap": True}, keep_alive=self.keep_alive)
        try:
            return np.array(res["embedding"]), 128
        except Exception as _e:
            log_exception(_e, res)
            raise Exception(f"Error: {res}")


class XinferenceEmbed(Base):
    """Xinference 嵌入模型封装。
    
    Xinference 是一个本地LLM推理框架，使用 OpenAI 兼容接口。
    """
    _FACTORY_NAME = "Xinference"

    def __init__(self, key, model_name="", base_url=""):
        """初始化 Xinference 嵌入客户端。
        
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: Xinference服务URL
        """
        base_url = urljoin(base_url, "v1")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def encode(self, texts: list):
        """批量编码文本为向量。"""
        batch_size = 16
        ress = []
        total_tokens = 0
        for i in range(0, len(texts), batch_size):
            res = None
            try:
                res = self.client.embeddings.create(input=texts[i : i + batch_size], model=self.model_name)
                ress.extend([d.embedding for d in res.data])
                total_tokens += total_token_count_from_response(res)
            except Exception as _e:
                log_exception(_e, res)
                raise Exception(f"Error: {res}")
        return np.array(ress), total_tokens

    def encode_queries(self, text):
        """编码单条查询文本。"""
        res = None
        try:
            res = self.client.embeddings.create(input=[text], model=self.model_name)
            return np.array(res.data[0].embedding), total_token_count_from_response(res)
        except Exception as _e:
            log_exception(_e, res)
            raise Exception(f"Error: {res}")


class YoudaoEmbed(Base):
    """有道嵌入模型封装。
    
    使用有道云服务提供向量嵌入能力，客户端通过类变量共享。
    """
    _FACTORY_NAME = "Youdao"
    _client = None  # 共享客户端实例

    def __init__(self, key=None, model_name="maidalun1020/bce-embedding-base_v1", **kwargs):
        """初始化有道嵌入客户端。
        
        Args:
            key: API密钥（可选）
            model_name: 模型名称
        """
        pass

    def encode(self, texts: list):
        """批量编码文本为向量。"""
        batch_size = 10
        res = []
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        for i in range(0, len(texts), batch_size):
            embds = YoudaoEmbed._client.encode(texts[i : i + batch_size])
            res.extend(embds)
        return np.array(res), token_count

    def encode_queries(self, text):
        """编码单条查询文本。"""
        embds = YoudaoEmbed._client.encode([text])
        return np.array(embds[0]), num_tokens_from_string(text)


class JinaMultiVecEmbed(Base):
    """Jina AI 多模态嵌入模型封装。
    
    支持文本和图片嵌入，使用 Jina AI 云服务。
    """
    _FACTORY_NAME = "Jina"

    def __init__(self, key, model_name="jina-embeddings-v4", base_url="https://api.jina.ai/v1/embeddings"):
        """初始化 Jina 嵌入客户端。
        
        Args:
            key: API密钥
            model_name: 模型名称（默认jina-embeddings-v4）
            base_url: API基础URL
        """
        self.base_url = "https://api.jina.ai/v1/embeddings"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        self.model_name = model_name

    def encode(self, texts: list[str | bytes], task="retrieval.passage"):
        """批量编码文本或图片为向量。
        
        支持文本和图片（base64编码）输入，v4模型支持多向量返回。
        
        Args:
            texts: 文本或图片字节列表
            task: 任务类型（retrieval.passage/retrieval.query等）
        """
        batch_size = 16
        ress = []
        token_count = 0
        input = []
        for text in texts:
            if isinstance(text, str):
                input.append({"text": text})
            elif isinstance(text, bytes):
                # 处理图片数据
                img_b64s = None
                try:
                    base64.b64decode(text, validate=True)
                    img_b64s = text.decode("utf8")
                except Exception:
                    img_b64s = base64.b64encode(text).decode("utf8")
                input.append({"image": img_b64s})
        for i in range(0, len(texts), batch_size):
            data = {"model": self.model_name, "input": input[i : i + batch_size]}
            # v4模型支持多向量
            if "v4" in self.model_name:
                data["return_multivector"] = True
            # v3/v4支持任务类型和自动截断
            if "v3" in self.model_name or "v4" in self.model_name:
                data["task"] = task
                data["truncate"] = True

            response = requests.post(self.base_url, headers=self.headers, json=data)
            try:
                res = response.json()
                for d in res["data"]:
                    if data.get("return_multivector", False):  # v4模型返回多向量
                        token_embs = np.asarray(d["embeddings"], dtype=np.float32)
                        chunk_emb = token_embs.mean(axis=0)  # 取平均作为chunk向量
                    else:
                        # 兼容 v2 / v3 两类返回格式
                        chunk_emb = np.asarray(d["embedding"], dtype=np.float32)
                    ress.append(chunk_emb)
                token_count += total_token_count_from_response(res)
            except Exception as _e:
                log_exception(_e, response)
                raise Exception(f"Error: {response}")
        return np.array(ress), token_count

    def encode_queries(self, text):
        """编码单条查询文本（使用retrieval.query任务）。"""
        embds, cnt = self.encode([text], task="retrieval.query")
        return np.array(embds[0]), cnt


class MistralEmbed(Base):
    """Mistral AI 嵌入模型封装。
    
    使用 Mistral AI 官方SDK调用嵌入服务，支持重试机制。
    """
    _FACTORY_NAME = "Mistral"

    def __init__(self, key, model_name="mistral-embed", base_url=None):
        """初始化 Mistral 嵌入客户端。
        
        Args:
            key: API密钥
            model_name: 模型名称（默认mistral-embed）
            base_url: API基础URL（可选）
        """
        from mistralai.client import MistralClient
        self.client = MistralClient(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list):
        """批量编码文本为向量。
        
        支持重试机制，最多重试5次，每次延迟20-60秒（随机）。
        """
        import time
        import random

        texts = [truncate(t, 8196) for t in texts]
        batch_size = 16
        ress = []
        token_count = 0
        for i in range(0, len(texts), batch_size):
            retry_max = 5
            while retry_max > 0:
                try:
                    res = self.client.embeddings(input=texts[i : i + batch_size], model=self.model_name)
                    ress.extend([d.embedding for d in res.data])
                    token_count += total_token_count_from_response(res)
                    break
                except Exception as _e:
                    if retry_max == 1:
                        log_exception(_e)
                    delay = random.uniform(20, 60)
                    time.sleep(delay)
                    retry_max -= 1
        return np.array(ress), token_count

    def encode_queries(self, text):
        """编码单条查询文本（支持重试）。"""
        import time
        import random

        retry_max = 5
        while retry_max > 0:
            try:
                res = self.client.embeddings(input=[truncate(text, 8196)], model=self.model_name)
                return np.array(res.data[0].embedding), total_token_count_from_response(res)
            except Exception as _e:
                if retry_max == 1:
                    log_exception(_e)
                delay = random.randint(20, 60)
                time.sleep(delay)
                retry_max -= 1


class BedrockEmbed(Base):
    _FACTORY_NAME = "Bedrock"

    def __init__(self, key, model_name, **kwargs):
        import boto3

        # `key` 字段实际存的是 JSON 字符串，需要先解码成 dict。
        # 必填项：`auth_mode`、`bedrock_region`。
        # 支持三种认证模式：
        # - `access_key_secret`：要求同时提供 `bedrock_ak` 和 `bedrock_sk`
        # - `iam_role`：要求提供 `aws_role_arn`，通过 STS 假设角色
        # - 其他情况：视作 `assume_role`，走默认 AWS 凭证链
        key = json.loads(key)
        mode = key.get("auth_mode")
        if not mode:
            logging.error("Bedrock auth_mode is not provided in the key")
            raise ValueError("Bedrock auth_mode must be provided in the key")

        self.bedrock_region = key.get("bedrock_region")

        self.model_name = model_name
        self.is_amazon = self.model_name.split(".")[0] == "amazon"
        self.is_cohere = self.model_name.split(".")[0] == "cohere"

        if mode == "access_key_secret":
            self.bedrock_ak = key.get("bedrock_ak")
            self.bedrock_sk = key.get("bedrock_sk")
            self.client = boto3.client(service_name="bedrock-runtime", region_name=self.bedrock_region, aws_access_key_id=self.bedrock_ak, aws_secret_access_key=self.bedrock_sk)
        elif mode == "iam_role":
            self.aws_role_arn = key.get("aws_role_arn")
            sts_client = boto3.client("sts", region_name=self.bedrock_region)
            resp = sts_client.assume_role(RoleArn=self.aws_role_arn, RoleSessionName="BedrockSession")
            creds = resp["Credentials"]

            self.client = boto3.client(
                service_name="bedrock-runtime",
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            )
        else:  # assume_role
            self.client = boto3.client("bedrock-runtime", region_name=self.bedrock_region)

    def encode(self, texts: list):
        texts = [truncate(t, 8196) for t in texts]
        embeddings = []
        token_count = 0
        for text in texts:
            if self.is_amazon:
                body = {"inputText": text}
            elif self.is_cohere:
                body = {"texts": [text], "input_type": "search_document"}

            response = self.client.invoke_model(modelId=self.model_name, body=json.dumps(body))
            try:
                model_response = json.loads(response["body"].read())
                embeddings.extend([model_response["embedding"]])
                token_count += num_tokens_from_string(text)
            except Exception as _e:
                log_exception(_e, response)

        return np.array(embeddings), token_count

    def encode_queries(self, text):
        embeddings = []
        token_count = num_tokens_from_string(text)
        if self.is_amazon:
            body = {"inputText": truncate(text, 8196)}
        elif self.is_cohere:
            body = {"texts": [truncate(text, 8196)], "input_type": "search_query"}

        response = self.client.invoke_model(modelId=self.model_name, body=json.dumps(body))
        try:
            model_response = json.loads(response["body"].read())
            embeddings.extend(model_response["embedding"])
        except Exception as _e:
            log_exception(_e, response)

        return np.array(embeddings), token_count


class GeminiEmbed(Base):
    _FACTORY_NAME = "Gemini"

    def __init__(self, key, model_name="gemini-embedding-001", **kwargs):
        from google import genai
        from google.genai import types

        self.key = key
        self.model_name = model_name[7:] if model_name.startswith("models/") else model_name
        self.client = genai.Client(api_key=self.key)
        self.types = types

    @staticmethod
    def _parse_embedding_vector(embedding):
        if isinstance(embedding, dict):
            values = embedding.get("values")
            if values is None:
                values = embedding.get("embedding")
            if values is not None:
                return values

        values = getattr(embedding, "values", None)
        if values is None:
            values = getattr(embedding, "embedding", None)
        if values is not None:
            return values

        raise TypeError(f"Unsupported embedding payload: {type(embedding)}")

    @classmethod
    def _parse_embedding_response(cls, response):
        if response is None:
            raise ValueError("Embedding response is empty")

        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")

        if embeddings is None:
            return [cls._parse_embedding_vector(response)]

        return [cls._parse_embedding_vector(item) for item in embeddings]

    def _build_embedding_config(self):
        task_type = "RETRIEVAL_DOCUMENT"
        if hasattr(self.types, "TaskType"):
            task_type = getattr(self.types.TaskType, "RETRIEVAL_DOCUMENT", task_type)
        try:
            return self.types.EmbedContentConfig(task_type=task_type, title="Embedding of single string")
        except TypeError:
        # 有些旧版 SDK 不接受 `title` 字段，这里做一次兼容降级。
            return self.types.EmbedContentConfig(task_type=task_type)

    def encode(self, texts: list):
        texts = [truncate(t, 2048) for t in texts]
        token_count = sum(num_tokens_from_string(text) for text in texts)
        config = self._build_embedding_config()
        batch_size = 16
        ress = []
        for i in range(0, len(texts), batch_size):
            result = None
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=texts[i : i + batch_size],
                    config=config,
                )
                ress.extend(self._parse_embedding_response(result))
            except Exception as _e:
                log_exception(_e, result)
                raise Exception(f"Error: {result}")
        return np.array(ress), token_count

    def encode_queries(self, text):
        config = self._build_embedding_config()
        result = None
        token_count = num_tokens_from_string(text)
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=[truncate(text, 2048)],
                config=config,
            )
            return np.array(self._parse_embedding_response(result)[0]), token_count
        except Exception as _e:
            log_exception(_e, result)
            raise Exception(f"Error: {result}")


class NvidiaEmbed(Base):
    _FACTORY_NAME = "NVIDIA"

    def __init__(self, key, model_name, base_url="https://integrate.api.nvidia.com/v1/embeddings"):
        if not base_url:
            base_url = "https://integrate.api.nvidia.com/v1/embeddings"
        self.api_key = key
        self.base_url = base_url
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }
        self.model_name = model_name
        if model_name == "nvidia/embed-qa-4":
            self.base_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings"
            self.model_name = "NV-Embed-QA"
        if model_name == "snowflake/arctic-embed-l":
            self.base_url = "https://ai.api.nvidia.com/v1/retrieval/snowflake/arctic-embed-l/embeddings"

    def encode(self, texts: list):
        batch_size = 16
        ress = []
        token_count = 0
        for i in range(0, len(texts), batch_size):
            payload = {
                "input": texts[i : i + batch_size],
                "input_type": "query",
                "model": self.model_name,
                "encoding_format": "float",
                "truncate": "END",
            }
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            try:
                res = response.json()
                ress.extend([d["embedding"] for d in res["data"]])
                token_count += total_token_count_from_response(res)
            except Exception as _e:
                log_exception(_e, response)
                raise Exception(f"Error: {response}")
        return np.array(ress), token_count

    def encode_queries(self, text):
        embds, cnt = self.encode([text])
        return np.array(embds[0]), cnt


class LmStudioEmbed(LocalAIEmbed):
    _FACTORY_NAME = "LM-Studio"

    def __init__(self, key, model_name, base_url):
        if not base_url:
            raise ValueError("Local llm url cannot be None")
        base_url = urljoin(base_url, "v1")
        self.client = OpenAI(api_key="lm-studio", base_url=base_url)
        self.model_name = model_name


class OpenAI_APIEmbed(OpenAIEmbed):
    _FACTORY_NAME = ["VLLM", "OpenAI-API-Compatible"]

    def __init__(self, key, model_name, base_url):
        if not base_url:
            raise ValueError("url cannot be None")
        base_url = urljoin(base_url, "v1")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name.split("___")[0]


class CoHereEmbed(Base):
    _FACTORY_NAME = "Cohere"

    def __init__(self, key, model_name, base_url=None):
        from cohere import Client

        self.client = Client(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list):
        batch_size = 16
        ress = []
        token_count = 0
        for i in range(0, len(texts), batch_size):
            res = self.client.embed(
                texts=texts[i : i + batch_size],
                model=self.model_name,
                input_type="search_document",
                embedding_types=["float"],
            )
            try:
                ress.extend([d for d in res.embeddings.float])
                token_count += total_token_count_from_response(res)
            except Exception as _e:
                log_exception(_e, res)
                raise Exception(f"Error: {res}")
        return np.array(ress), token_count

    def encode_queries(self, text):
        res = self.client.embed(
            texts=[text],
            model=self.model_name,
            input_type="search_query",
            embedding_types=["float"],
        )
        try:
            return np.array(res.embeddings.float[0]), int(total_token_count_from_response(res))
        except Exception as _e:
            log_exception(_e, res)
            raise Exception(f"Error: {res}")


class TogetherAIEmbed(OpenAIEmbed):
    _FACTORY_NAME = "TogetherAI"

    def __init__(self, key, model_name, base_url="https://api.together.xyz/v1"):
        if not base_url:
            base_url = "https://api.together.xyz/v1"
        super().__init__(key, model_name, base_url=base_url)


class PerfXCloudEmbed(OpenAIEmbed):
    _FACTORY_NAME = "PerfXCloud"

    def __init__(self, key, model_name, base_url="https://cloud.perfxlab.cn/v1"):
        if not base_url:
            base_url = "https://cloud.perfxlab.cn/v1"
        super().__init__(key, model_name, base_url)


class UpstageEmbed(OpenAIEmbed):
    _FACTORY_NAME = "Upstage"

    def __init__(self, key, model_name, base_url="https://api.upstage.ai/v1/solar"):
        if not base_url:
            base_url = "https://api.upstage.ai/v1/solar"
        super().__init__(key, model_name, base_url)


class SILICONFLOWEmbed(Base):
    _FACTORY_NAME = "SILICONFLOW"

    def __init__(self, key, model_name, base_url="https://api.siliconflow.cn/v1/embeddings"):
        normalized_base_url = (base_url or "").strip()
        if not normalized_base_url:
            normalized_base_url = "https://api.siliconflow.cn/v1/embeddings"
        if "/embeddings" not in normalized_base_url:
            normalized_base_url = urljoin(f"{normalized_base_url.rstrip('/')}/", "embeddings").rstrip("/")
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        }
        self.base_url = normalized_base_url
        self.model_name = model_name

    def encode(self, texts: list):
        batch_size = 16
        ress = []
        token_count = 0
        for i in range(0, len(texts), batch_size):
            texts_batch = texts[i : i + batch_size]
            if self.model_name in ["BAAI/bge-large-zh-v1.5", "BAAI/bge-large-en-v1.5"]:
                # limit 512, 340 is almost safe
                texts_batch = [" " if not text.strip() else truncate(text, 256) for text in texts_batch]
            else:
                texts_batch = [" " if not text.strip() else text for text in texts_batch]

            payload = {
                "model": self.model_name,
                "input": texts_batch,
                "encoding_format": "float",
            }
            response = requests.post(self.base_url, json=payload, headers=self.headers)
            try:
                res = response.json()
                ress.extend([d["embedding"] for d in res["data"]])
                token_count += total_token_count_from_response(res)
            except Exception as _e:
                log_exception(_e, response)
                raise Exception(f"Error: {response}")

        return np.array(ress), token_count

    def encode_queries(self, text):
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float",
        }
        response = requests.post(self.base_url, json=payload, headers=self.headers)
        try:
            res = response.json()
            return np.array(res["data"][0]["embedding"]), total_token_count_from_response(res)
        except Exception as _e:
            log_exception(_e, response)
            raise Exception(f"Error: {response}")


class ReplicateEmbed(Base):
    _FACTORY_NAME = "Replicate"

    def __init__(self, key, model_name, base_url=None):
        from replicate.client import Client

        self.model_name = model_name
        self.client = Client(api_token=key)

    def encode(self, texts: list):
        batch_size = 16
        token_count = sum([num_tokens_from_string(text) for text in texts])
        ress = []
        for i in range(0, len(texts), batch_size):
            res = self.client.run(self.model_name, input={"texts": texts[i : i + batch_size]})
            ress.extend(res)
        return np.array(ress), token_count

    def encode_queries(self, text):
        res = self.client.embed(self.model_name, input={"texts": [text]})
        return np.array(res), num_tokens_from_string(text)


class BaiduYiyanEmbed(Base):
    _FACTORY_NAME = "BaiduYiyan"

    def __init__(self, key, model_name, base_url=None):
        import qianfan

        key = json.loads(key)
        ak = key.get("yiyan_ak", "")
        sk = key.get("yiyan_sk", "")
        self.client = qianfan.Embedding(ak=ak, sk=sk)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=16):
        res = self.client.do(model=self.model_name, texts=texts).body
        try:
            return (
                np.array([r["embedding"] for r in res["data"]]),
                total_token_count_from_response(res),
            )
        except Exception as _e:
            log_exception(_e, res)
            raise Exception(f"Error: {res}")

    def encode_queries(self, text):
        res = self.client.do(model=self.model_name, texts=[text]).body
        try:
            return (
                np.array([r["embedding"] for r in res["data"]]),
                total_token_count_from_response(res),
            )
        except Exception as _e:
            log_exception(_e, res)
            raise Exception(f"Error: {res}")


class VoyageEmbed(Base):
    _FACTORY_NAME = "Voyage AI"

    def __init__(self, key, model_name, base_url=None):
        import voyageai

        self.client = voyageai.Client(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list):
        batch_size = 16
        ress = []
        token_count = 0
        for i in range(0, len(texts), batch_size):
            res = self.client.embed(texts=texts[i : i + batch_size], model=self.model_name, input_type="document")
            try:
                ress.extend(res.embeddings)
                token_count += res.total_tokens
            except Exception as _e:
                log_exception(_e, res)
                raise Exception(f"Error: {res}")
        return np.array(ress), token_count

    def encode_queries(self, text):
        res = self.client.embed(texts=text, model=self.model_name, input_type="query")
        try:
            return np.array(res.embeddings)[0], res.total_tokens
        except Exception as _e:
            log_exception(_e, res)
            raise Exception(f"Error: {res}")


class HuggingFaceEmbed(Base):
    _FACTORY_NAME = "HuggingFace"

    def __init__(self, key, model_name, base_url=None, **kwargs):
        if not model_name:
            raise ValueError("Model name cannot be None")
        self.key = key
        self.model_name = model_name.split("___")[0]
        self.base_url = base_url or "http://127.0.0.1:8080"

    def encode(self, texts: list):
        response = requests.post(f"{self.base_url}/embed", json={"inputs": texts}, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            embeddings = response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        return np.array(embeddings), sum([num_tokens_from_string(text) for text in texts])

    def encode_queries(self, text: str):
        response = requests.post(f"{self.base_url}/embed", json={"inputs": text}, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            embedding = response.json()[0]
            return np.array(embedding), num_tokens_from_string(text)
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")


class VolcEngineEmbed(Base):
    _FACTORY_NAME = "VolcEngine"

    def __init__(self, key, model_name, base_url="https://ark.cn-beijing.volces.com/api/v3"):
        if not base_url:
            base_url = "https://ark.cn-beijing.volces.com/api/v3"
        self.base_url = base_url

        cfg = json.loads(key)
        self.ark_api_key = cfg.get("ark_api_key", "")
        self.model_name = model_name

    @staticmethod
    def _extract_embedding(result: dict) -> list[float]:
        if not isinstance(result, dict):
            raise TypeError(f"Unexpected response type: {type(result)}")

        data = result.get("data")
        if data is None:
            raise KeyError("Missing 'data' in response")

        if isinstance(data, list):
            if not data:
                raise ValueError("Empty 'data' in response")
            item = data[0]
        elif isinstance(data, dict):
            item = data
        else:
            raise TypeError(f"Unexpected 'data' type: {type(data)}")

        if not isinstance(item, dict):
            raise TypeError("Unexpected item shape in 'data'")
        if "embedding" not in item:
            raise KeyError("Missing 'embedding' in response item")
        return item["embedding"]

    def _encode_texts(self, texts: list[str]):
        from common.http_client import sync_request

        url = f"{self.base_url}/embeddings/multimodal"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.ark_api_key}"}

        ress: list[list[float]] = []
        total_tokens = 0
        for text in texts:
            request_body = {"model": self.model_name, "input": [{"type": "text", "text": text}]}
            response = sync_request(method="POST", url=url, headers=headers, json=request_body, timeout=60)
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code} - {response.text}")
            result = response.json()
            try:
                ress.append(self._extract_embedding(result))
                total_tokens += total_token_count_from_response(result)
            except Exception as _e:
                log_exception(_e)

        return np.array(ress), total_tokens

    def encode(self, texts: list):
        return self._encode_texts(texts)

    def encode_queries(self, text: str):
        embeddings, tokens = self._encode_texts([text])
        return embeddings[0], tokens


class GPUStackEmbed(OpenAIEmbed):
    _FACTORY_NAME = "GPUStack"

    def __init__(self, key, model_name, base_url):
        if not base_url:
            raise ValueError("url cannot be None")
        base_url = urljoin(base_url, "v1")

        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name


class NovitaEmbed(SILICONFLOWEmbed):
    _FACTORY_NAME = "NovitaAI"

    def __init__(self, key, model_name, base_url="https://api.novita.ai/v3/openai/embeddings"):
        if not base_url:
            base_url = "https://api.novita.ai/v3/openai/embeddings"
        super().__init__(key, model_name, base_url)


class GiteeEmbed(SILICONFLOWEmbed):
    _FACTORY_NAME = "GiteeAI"

    def __init__(self, key, model_name, base_url="https://ai.gitee.com/v1/embeddings"):
        if not base_url:
            base_url = "https://ai.gitee.com/v1/embeddings"
        super().__init__(key, model_name, base_url)


class DeepInfraEmbed(OpenAIEmbed):
    _FACTORY_NAME = "DeepInfra"

    def __init__(self, key, model_name, base_url="https://api.deepinfra.com/v1/openai"):
        if not base_url:
            base_url = "https://api.deepinfra.com/v1/openai"
        super().__init__(key, model_name, base_url)


class Ai302Embed(Base):
    _FACTORY_NAME = "302.AI"

    def __init__(self, key, model_name, base_url="https://api.302.ai/v1/embeddings"):
        if not base_url:
            base_url = "https://api.302.ai/v1/embeddings"
        super().__init__(key, model_name, base_url)


class CometAPIEmbed(OpenAIEmbed):
    _FACTORY_NAME = "CometAPI"

    def __init__(self, key, model_name, base_url="https://api.cometapi.com/v1"):
        if not base_url:
            base_url = "https://api.cometapi.com/v1"
        super().__init__(key, model_name, base_url)


class DeerAPIEmbed(OpenAIEmbed):
    _FACTORY_NAME = "DeerAPI"

    def __init__(self, key, model_name, base_url="https://api.deerapi.com/v1"):
        if not base_url:
            base_url = "https://api.deerapi.com/v1"
        super().__init__(key, model_name, base_url)


class JiekouAIEmbed(OpenAIEmbed):
    _FACTORY_NAME = "Jiekou.AI"

    def __init__(self, key, model_name, base_url="https://api.jiekou.ai/openai/v1/embeddings"):
        if not base_url:
            base_url = "https://api.jiekou.ai/openai/v1/embeddings"
        super().__init__(key, model_name, base_url)


class RAGconEmbed(OpenAIEmbed):
    """
    RAGcon Embedding Provider - routes through LiteLLM proxy

    Default Base URL: https://connect.ragcon.ai/v1
    """

    _FACTORY_NAME = "RAGcon"

    def __init__(self, key, model_name="text-embedding-3-small", base_url=None):
        if not base_url:
            base_url = "https://connect.ragcon.com/v1"

        super().__init__(key, model_name, base_url)


class PerplexityEmbed(Base):
    _FACTORY_NAME = "Perplexity"

    def __init__(self, key, model_name="pplx-embed-v1-0.6b", base_url="https://api.perplexity.ai"):
        if not base_url:
            base_url = "https://api.perplexity.ai"
        self.base_url = base_url.rstrip("/")
        self.api_key = key
        self.model_name = model_name
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _decode_base64_int8(b64_str):
        raw = base64.b64decode(b64_str)
        return np.frombuffer(raw, dtype=np.int8).astype(np.float32)

    def _is_contextualized(self):
        return "context" in self.model_name

    def encode(self, texts: list):
        batch_size = 512
        ress = []
        token_count = 0

        if self._is_contextualized():
            url = f"{self.base_url}/v1/contextualizedembeddings"
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                payload = {
                    "model": self.model_name,
                    "input": [[chunk] for chunk in batch],
                    "encoding_format": "base64_int8",
                }
                response = requests.post(url, headers=self.headers, json=payload)
                try:
                    res = response.json()
                    for doc in res["data"]:
                        for chunk_emb in doc["data"]:
                            ress.append(self._decode_base64_int8(chunk_emb["embedding"]))
                    token_count += res.get("usage", {}).get("total_tokens", 0)
                except Exception as _e:
                    log_exception(_e, response)
                    raise Exception(f"Error: {response.text}")
        else:
            url = f"{self.base_url}/v1/embeddings"
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                payload = {
                    "model": self.model_name,
                    "input": batch,
                    "encoding_format": "base64_int8",
                }
                response = requests.post(url, headers=self.headers, json=payload)
                try:
                    res = response.json()
                    for d in res["data"]:
                        ress.append(self._decode_base64_int8(d["embedding"]))
                    token_count += res.get("usage", {}).get("total_tokens", 0)
                except Exception as _e:
                    log_exception(_e, response)
                    raise Exception(f"Error: {response.text}")

        return np.array(ress), token_count

    def encode_queries(self, text):
        embds, cnt = self.encode([text])
        return np.array(embds[0]), cnt