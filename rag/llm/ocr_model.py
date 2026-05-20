"""OCR / 文档解析模型适配层。

目前主要承接 MinerU 这类 PDF 解析能力，并统一成 `parse_pdf()` 接口，
供上层文档处理流程直接调用。

支持的解析器：
- MinerU：基于 MinerU 的 PDF 解析器，支持多种解析方法
"""

#

import json
import logging
import os


class Base:
    """OCR / 文档解析模型基类。
    
    定义所有OCR/文档解析模型必须实现的接口。
    """

    def __init__(self, key="", model_name="", lang="Chinese", base_url="", **kwargs):
        """初始化OCR模型。
        
        Args:
            key: API密钥或配置JSON字符串
            model_name: 模型名称
            lang: 语言（默认Chinese）
            base_url: 服务URL
        """
        self.model_name = model_name
        self.lang = lang
        self.base_url = base_url

    def parse_pdf(self, filepath=None, binary=None, callback=None, **kwargs):
        """解析PDF文件。
        
        Args:
            filepath: PDF文件路径
            binary: PDF文件二进制内容
            callback: 进度回调函数
            
        Returns:
            解析结果
        """
        raise NotImplementedError


class MinerU(Base):
    """基于 MinerU 的 PDF 解析器。
    
    MinerU 是一个专业的PDF解析工具，支持多种解析方法（auto、ocr、text等），
    能够提取文本、表格、图片等内容，并保持良好的文档结构。
    """

    _FACTORY_NAME = "MinerU"

    def __init__(self, key="", model_name="", lang="Chinese", base_url="", **kwargs):
        """初始化 MinerU PDF 解析器。
        
        配置优先级：key中的JSON配置 > 环境变量 > 默认配置
        
        Args:
            key: JSON格式的配置字符串，包含MINERU相关参数
            model_name: 模型名称
            lang: 语言（默认Chinese）
            base_url: 服务URL
        """
        super().__init__(key=key, model_name=model_name, lang=lang, base_url=base_url, **kwargs)

        cfg: dict = {}
        if key:
            try:
                cfg = json.loads(key)
            except Exception:
                logging.warning("MinerU: failed to parse config JSON from key field: %r", key)

        from common.constants import MINERU_DEFAULT_CONFIG

        def _get(k: str):
            """获取配置值，优先级：cfg > 环境变量 > 默认配置"""
            return cfg.get(k) or os.environ.get(k, MINERU_DEFAULT_CONFIG[k])

        self.api_url = _get("MINERU_APISERVER")      # API服务器地址
        self.server_url = _get("MINERU_SERVER_URL")  # 服务地址
        self.output_dir = _get("MINERU_OUTPUT_DIR")  # 输出目录
        self.backend = _get("MINERU_BACKEND")        # 后端类型
        self.delete_output = bool(int(_get("MINERU_DELETE_OUTPUT") or 1))  # 是否删除临时输出

    def parse_pdf(self, filepath=None, binary=None, callback=None,
                  parse_method="auto", lang=None, **kwargs):
        """调用 MinerU 执行实际 PDF 解析。
        
        Args:
            filepath: PDF文件路径（与binary二选一）
            binary: PDF文件二进制内容（与filepath二选一）
            callback: 进度回调函数
            parse_method: 解析方法（auto/ocr/text等）
            lang: 语言（覆盖默认值）
            
        Returns:
            dict: 解析结果，包含文本、表格、图片等信息
        """
        from parser.mineru_parser import MinerUPdfParser

        parser = MinerUPdfParser(
            api_url=self.api_url,
            server_url=self.server_url,
            output_dir=self.output_dir,
            backend=self.backend,
            delete_output=self.delete_output,
        )
        return parser.parse_pdf(
            filepath=filepath,
            binary=binary,
            callback=callback,
            parse_method=parse_method,
            lang=lang or self.lang,
            **kwargs,
        )