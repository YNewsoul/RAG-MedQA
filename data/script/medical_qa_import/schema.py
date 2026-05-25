"""共享常量、数据结构与纯函数工具。

这个模块只保留“无副作用”的基础能力：
- 常量
- dataclass
- 文本规范化
- 分片名/ID 的稳定生成

这样做的好处是：
- `planning`、`importing`、`reporting` 都能复用
- 不会把数据库、ES、embedding 等重依赖混进基础层
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import xxhash

from .bootstrap import PROJECT_ROOT


# 连续空白压成一个空格，避免换行、制表符、多空格影响去重和检索。
WHITESPACE_RE = re.compile(r"\s+")

# department 只有满足“像一个正常科室名”的最基本规则，才允许参与独立分桶。
# 这样可以过滤掉误写进 department 字段的回答片段、病史描述等脏值。
CLEAN_DEPARTMENT_RE = re.compile(r"^[\u4e00-\u9fa5A-Za-z]+$")

# 默认输入输出位置。
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "medical" / "qa"
DEFAULT_WORKDIR = PROJECT_ROOT / "data" / "script" / "output" / "medical_qa_import"

# 支持的运行阶段。
# 这里只保留两个真正需要暴露给用户的阶段：
# 1. materialize：做数据扫描、规划和 JSONL 物化
# 2. import：消费既有 JSONL，把数据正式写入库和 ES
# 同时保留 all，便于首次导入时一条命令跑完。
PHASES = {"all", "materialize", "import"}

# 文件名前缀所代表的一级医学大类。
MAJOR_ORDER = ["儿科", "内科", "外科", "妇产科", "男科", "肿瘤科"]

# 独立成桶阈值的覆盖规则。
# 默认大类用 2000；男科和肿瘤科病种粒度更细，阈值适当放低。
THRESHOLD_OVERRIDES = {
    "男科": 1500,
    "肿瘤科": 1000,
}

# 每个 logical document 控制在约 8000 条 QA。
SHARD_TARGET_COUNT = 8000


@dataclass
class FileStats:
    """单个原始 JSON 文件的扫描统计结果。"""

    filename: str
    major_category: str
    size_bytes: int
    total_rows: int = 0
    invalid_rows: int = 0
    local_duplicate_rows: int = 0
    global_duplicate_rows: int = 0
    kept_rows: int = 0
    clean_department_rows: int = 0
    dirty_department_rows: int = 0
    unique_departments: int = 0
    top_departments: list[tuple[str, int]] | None = None
    dirty_department_samples: list[str] | None = None


@dataclass
class ShardPlan:
    """一个 logical document 的规划结果。"""

    major_category: str  # 一级医学大类
    source_file: str  # 原始 JSON 文件名
    bucket_department: str  # 科室名
    bucket_type: str  # 分桶类型，如 "department" 或 "major_category"
    part_index: int  # 分片索引
    shard_name: str  # 分片名
    planned_count: int  # 计划 QA 数量
    shard_file: str  # 分片 JSONL 文件名
    doc_id: str = ""  # 稳定的 document id
    file_md5: str = ""  # 文件的 MD5 值
    file_size: int = 0  # 文件大小，单位字节


def normalize_text(value: str | None) -> str:
    """对文本做轻量规范化。

    这里只做最基础的清洗：
    - None -> ""
    - 连续空白折叠
    - 去首尾空白

    这样既能提升去重稳定性，也不会破坏中文原句结构。
    """

    if value is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def major_from_filename(filename: str) -> str:
    """根据文件名前缀识别一级医学大类。"""

    for major in MAJOR_ORDER:
        if filename.startswith(major):
            return major
    return Path(filename).stem


def department_is_clean(value: str) -> bool:
    """判断 department 是否足够“像一个正常科室名”。

    这里不追求医学分类学上的绝对正确，而是做一个工程上可用的过滤：
    - 必须非空
    - 长度不要过长
    - 只能是中文或英文字母
    """

    value = normalize_text(value)
    return bool(value) and len(value) <= 12 and bool(CLEAN_DEPARTMENT_RE.fullmatch(value))


def threshold_for_major(major: str, default_threshold: int) -> int:
    """返回某个一级大类使用的独立分桶阈值。"""

    return THRESHOLD_OVERRIDES.get(major, default_threshold)


def dedupe_hash(ask: str, answer: str) -> str:
    """按 ask + answer 生成去重键。

    注意故意不把 title 放进哈希。
    原因是同一条问答可能会有不同标题改写，但问题和答案主体没变。
    """

    return hashlib.md5(f"{ask}\n{answer}".encode("utf-8")).hexdigest()


def shard_rel_path(shard_name: str) -> str:
    """把 shard 名映射成 workdir 内部的 JSONL 相对路径。"""

    return f"normalized_shards/{shard_name}.jsonl"


def deterministic_doc_id(kb_id: str, shard_name: str) -> str:
    """为 logical document 生成稳定的 document id。

    这样相同知识库里的相同 shard，每次 rerun 都会得到同一个 doc_id，
    便于：
    - 判断是否已经成功导入
    - 失败后按 shard 清理重建
    """

    return hashlib.md5(f"{kb_id}:{shard_name}".encode("utf-8")).hexdigest()


def deterministic_chunk_id(doc_id: str, ask: str, answer: str) -> str:
    """为单条 QA 生成稳定的 chunk id"""
    return xxhash.xxh64(f"{doc_id}\n{ask}\n{answer}".encode("utf-8")).hexdigest()
