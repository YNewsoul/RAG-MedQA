"""脚本本地文件读写工具。"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .schema import MinerUCacheInfo, ShardPlan


def ensure_parent(path: Path) -> None:
    """确保目标文件的父目录存在。

    这是所有写文件动作前的统一兜底，避免每个调用点都重复 mkdir。
    """

    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    """以 UTF-8 写文本。"""

    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    """以 UTF-8 写 JSON。

    这里统一使用：
    - `ensure_ascii=False` 保留中文
    - `indent=2` 便于人工阅读

    这对中间产物和报告排障都很重要。
    """

    ensure_parent(path)

    def _default(obj):
        # 允许直接把 dataclass 列表写入 JSON，
        # 这样上层不用先手动挨个 asdict。
        if is_dataclass(obj):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_default),
        encoding="utf-8",
    )


def read_json(path: Path, default: Any = None) -> Any:
    """读取 JSON；不存在时返回默认值。"""

    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """一次性写出 JSONL。

    materialize 阶段的标准中间产物就是 JSONL。
    import 阶段只消费这些 JSONL，不再关心原始 PDF。
    """

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 为字典列表。

    这里读取成整表列表是有意为之：
    当前 logical document 规模被设计得相对可控，
    换来的是 import 阶段逻辑更清晰。
    """

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_shard_plans_from_file(shard_plan_path: Path) -> list[ShardPlan]:
    """从 `shard_plan.json` 反序列化分片计划。

    import-only 模式强依赖这个文件。
    如果它不存在，说明 materialize 还没跑过，或者 workdir 不对。
    """

    raw = read_json(shard_plan_path, default=None)
    if raw is None:
        raise FileNotFoundError(
            f"Shard plan not found: {shard_plan_path}. Please run materialize first."
        )
    return [ShardPlan(**item) for item in raw]


def load_mineru_cache(cache_path: Path) -> dict[str, MinerUCacheInfo]:
    """读取 MinerU 预处理缓存索引。

    结构是：
    `source_name -> MinerUCacheInfo`
    这样单个 PDF 是否可复用，查起来最直观。
    """

    raw = read_json(cache_path, default={}) or {}
    return {name: MinerUCacheInfo(**item) for name, item in raw.items()}
