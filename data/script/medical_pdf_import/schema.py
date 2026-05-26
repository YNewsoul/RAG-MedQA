"""PDF 建库脚本的常量、数据结构和纯函数工具。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import xxhash

from .bootstrap import PROJECT_ROOT


# 默认输入目录：放医疗 PDF 原始资料。
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "medical" / "pdf"
# 默认工作目录：所有中间产物、缓存和报告都写到这里。
DEFAULT_WORKDIR = PROJECT_ROOT / "data" / "script" / "output" / "medical_pdf_import"
# 脚本支持的阶段。
PHASES = {"materialize", "import", "all"}

# 每个 chunk 的目标 token 数。
# PDF 比 QA 更偏长文本，通常可以适当放大。
DEFAULT_CHUNK_TOKEN_NUM = 768
# chunk 间默认不重叠；后续如果发现跨段衔接不够，可以调大。
DEFAULT_CHUNK_OVERLAP_TOKENS = 0
# 一个 logical document 的目标 token 预算。
# 它决定“一个 document 包含多少 chunk”，影响断点粒度和管理成本。
DEFAULT_LOGICAL_DOC_TOKEN_NUM = 6144
# 单个 logical document 最多允许多少个 chunk，防止某些章节异常膨胀。
DEFAULT_LOGICAL_DOC_MAX_CHUNKS = 40
# 用多少层标题路径定义“章节根”。
DEFAULT_HEADING_SPLIT_DEPTH = 2
# MinerU 默认先走 auto，让它自行判断文本层/OCR层。
DEFAULT_MINERU_PARSE_METHOD = "auto"


@dataclass(slots=True)
class PdfSourceInfo:
    """单个原始 PDF 的基础信息。

    这是最上游的“源文件视图”。
    后续无论是 MinerU 缓存、logical document 分片还是入库报告，
    都会围绕这份信息展开。
    """

    # 原始文件名，例如“临床诊疗指南 — 呼吸病学分册.pdf”
    source_name: str
    # 原始 PDF 的绝对路径或可解析路径
    source_path: str
    # 原始 PDF 的内容哈希，用于判断源文件是否发生变化
    source_md5: str
    # 文件大小（字节）
    file_size: int
    # 页数，用于评估复杂度、报告展示和后续切分判断
    page_count: int
    # 用于展示和 chunk 拼装的标题
    title: str
    # 从文件名推导出的专题/分册标签
    specialty: str
    # 基于 source_name 稳定生成的源文件 ID
    source_id: str


@dataclass(slots=True)
class MinerUCacheInfo:
    """单个 PDF 的 MinerU 预处理缓存信息。

    这层缓存的意义是：避免每次 materialize 都重新跑 MinerU。
    对 PDF 而言，MinerU 是整条链路里最重的步骤之一，因此缓存非常关键。
    """

    source_name: str
    source_md5: str
    # markdown / blocks 文件都相对 workdir 存储，便于目录整体迁移
    markdown_file: str
    block_file: str
    block_count: int
    table_count: int


@dataclass(slots=True)
class ShardPlan:
    """一个 logical document 分片的计划信息。

    可以把它理解成“未来要入库的一份 document 的蓝图”。
    它在 materialize 阶段生成，在 import 阶段被消费。
    """

    source_name: str
    source_md5: str
    page_count: int
    title: str
    specialty: str
    chapter_root: str
    section_path: str
    part_index: int
    shard_name: str
    shard_file: str
    planned_count: int
    token_estimate: int
    # 下面三个字段在 materialize 后期补齐，
    # import 阶段会拿它们做稳定入库和幂等校验。
    doc_id: str = ""
    file_md5: str = ""
    file_size: int = 0


def normalize_text(text: str | None) -> str:
    """做轻量文本清洗，避免分片与命名中混入过多噪声。

    注意这里只做“安全且保守”的规范化：
    - 全角空格转半角空格
    - 连续空白压缩
    - 两端 trim

    它不尝试做语义修正，因为那样容易误伤医学术语。
    """

    if text is None:
        return ""
    text = str(text).replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sanitize_name(text: str, max_len: int = 40) -> str:
    """把任意标题清洗成适合文件名/分片名的片段。

    这个函数只用于“命名”，不用于正文。
    所以它会比 `normalize_text()` 更激进：
    - 去掉 Windows 文件名非法字符
    - 去掉空格
    - 限制长度
    """

    text = normalize_text(text)
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return "untitled"
    return text[:max_len]


def infer_pdf_title(filename: str) -> str:
    """从文件名推导一个稳定、可展示的 PDF 标题。"""

    stem = Path(filename).stem
    stem = normalize_text(stem)
    return stem or filename


def infer_specialty(filename: str) -> str:
    """从文件名推导医学专题名。

    这里不追求百分之百语义完美，目标是得到一个足够稳定的分册/专科标签。
    """

    # 这里的目标不是做完美的 NER，而是从较稳定的文件命名里
    # 提取出能用于过滤/展示的专题标签。
    stem = infer_pdf_title(filename)
    stem = re.sub(r"^临床诊疗指南\s*[—_-]?\s*", "", stem)
    stem = re.sub(r"_[^_]+著$", "", stem)
    stem = re.sub(r"\d{4}$", "", stem)
    stem = stem.replace(".PDF", "").replace(".pdf", "")
    stem = stem.strip(" _-")
    if stem.endswith("分册"):
        stem = stem[:-2]
    stem = normalize_text(stem)
    return stem or "未分类专题"


def heading_path_to_text(heading_path: list[str]) -> str:
    """把标题层级路径转成可展示字符串。"""

    cleaned = [normalize_text(item) for item in heading_path if normalize_text(item)]
    return " > ".join(cleaned)


def deterministic_source_id(source_name: str) -> str:
    """为原始 PDF 生成稳定 source_id。

    这里故意只用 `source_name`，而不把 md5 拼进来。
    因为 source_id 更像“文件身份”，而 source_md5 更像“文件版本”。
    """

    return hashlib.md5(source_name.encode("utf-8")).hexdigest()


def deterministic_doc_id(kb_id: str, shard_name: str) -> str:
    """为 logical document 生成稳定 document id。

    这样同一个 KB 下的同一个分片，无论 rerun 多少次，
    只要 `kb_id + shard_name` 不变，它的 `doc_id` 就不会变。
    这是 import 阶段做“已完成跳过 / 半成品清理重建”的前提。
    """

    return hashlib.md5(f"{kb_id}:{shard_name}".encode("utf-8")).hexdigest()


def deterministic_chunk_id(doc_id: str, row_index: int, content_with_weight: str) -> str:
    """为单个 chunk 生成稳定 chunk id。

    这里使用：
    - 所属 document
    - 在 document 内的顺序
    - 最终语义文本

    三者共同决定一个 chunk 的身份。
    """

    return xxhash.xxh64(
        f"{doc_id}\n{row_index}\n{content_with_weight}".encode("utf-8")
    ).hexdigest()


def file_md5(path: Path) -> str:
    """计算文件 MD5。

    采用流式读取，避免一次性把大 PDF 或大 JSONL 全部读进内存。
    """

    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_shard_name(source: PdfSourceInfo, chapter_root: str, part_index: int) -> str:
    """生成逻辑文档分片名。

    命名形态示例：
    `pdf_呼吸病学_第一章总论_p001`

    这个名字会同时用于：
    - 物化 JSONL 文件名
    - document.name
    - 报告展示
    """

    specialty = sanitize_name(source.specialty, 24)
    chapter = sanitize_name(chapter_root or "正文", 24)
    return f"pdf_{specialty}_{chapter}_p{part_index:03d}"
