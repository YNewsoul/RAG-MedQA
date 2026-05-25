"""与本地文件读写相关的辅助函数。"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import ShardPlan


def load_json_records(path: Path, limit: int | None = None) -> list[dict]:
    """读取单个原始 JSON 文件。"""

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array.")
    if limit is not None:
        return data[:limit]
    return data


def load_jsonl(path: Path) -> list[dict]:
    """读取标准化 JSONL 分片。

    import 阶段不再关心原始大 JSON 是什么结构，
    它只消费这里的标准化 JSONL。
    """

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict | list) -> None:
    """把对象按 UTF-8 JSON 形式写盘。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, content: str) -> None:
    """把文本按 UTF-8 写盘。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_shard_plans_from_file(shard_plan_path: Path) -> list[ShardPlan]:
    """从 workdir 中加载已经生成好的 shard 计划。

    这个辅助函数主要服务于 `--phase import`：
    - import 不会重新扫描原始数据
    - 它只依赖之前 materialize 阶段留下的 `shard_plan.json`

    如果用户直接跑 import，但 workdir 里还没有 `shard_plan.json`，
    这里会给出一个明确、可操作的报错，而不是让后面的流程在更深层才失败。
    """

    if not shard_plan_path.exists():
        raise FileNotFoundError(
            f"Missing shard plan: {shard_plan_path}. "
            "Please run `--phase materialize` or `--phase all` first."
        )
    return [ShardPlan(**item) for item in json.loads(shard_plan_path.read_text(encoding="utf-8"))]
