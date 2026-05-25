"""命令行入口实现。

这个模块只负责组织流程，不直接承载核心业务细节。
这样以后如果你想：
- 加一个 GUI 入口
- 改成 notebook 调用
- 或者给脚本再套一层任务调度

都只需要复用这里的主流程，而不用再碰底层 planning / importing 细节。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from .importing import ensure_kb_folder, ensure_knowledge_base, ensure_runtime, import_shards
from .io_utils import load_shard_plans_from_file, write_json, write_text
from .planning import materialize_shards, scan_corpus
from .reporting import markdown_report
from .schema import DEFAULT_DATA_DIR, DEFAULT_WORKDIR, PHASES


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    参数设计遵循两个原则：

    1. materialize 和 import 解耦
       允许先把分片中间产物准备好，再选择合适时机入库。
    2. rerun 尽量简单
       发生断网或中断后，通常只需要保留同一个 kb-name 和 workdir 重新执行即可。
    """

    parser = argparse.ArgumentParser(description="Import medical QA corpus into a QA knowledge base.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="QA JSON directory.")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR), help="Output and checkpoint directory.")
    parser.add_argument("--kb-name", default="medical_qa_kb", help="Knowledge base name.")
    parser.add_argument(
        "--phase",
        default="all",
        choices=sorted(PHASES),
        help="Run materialize/import/all. Note: materialize includes internal scan+plan.",
    )
    parser.add_argument("--limit-per-file", type=int, default=None, help="Only process the first N rows of each JSON file.")
    parser.add_argument("--default-threshold", type=int, default=2000, help="Department split threshold for most categories.")
    parser.add_argument("--embed-batch-size", type=int, default=512, help="Embedding batch size.")
    parser.add_argument("--insert-batch-size", type=int, default=256, help="ES insert batch size.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    """脚本主入口

    - materialize：扫描原始 QA，并生成 shard_plan + JSONL 分片
    - import：读取既有 shard_plan + JSONL，正式写库
    - all：先 materialize，再 import

    其中需要特别注意：
    - import-only 不会重新扫描原始数据
    - import-only 强依赖 workdir 里已经存在 `shard_plan.json`
    """

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    
    # 路径管理
    data_dir = Path(args.data_dir).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    shard_plan_path = workdir / "shard_plan.json"
    report_path = workdir / "reports" / "qa_import_report.md"
    analysis: dict | None = None

    # materialize 阶段承担三件事：
    # 1. 扫描原始目录
    # 2. 生成稳定分片计划
    # 3. 物化出标准化 JSONL
    if args.phase in {"all", "materialize"}:
        analysis, shard_plans = scan_corpus(data_dir, args.limit_per_file, args.default_threshold)
        shard_plans = materialize_shards(
            data_dir=data_dir,
            workdir=workdir,
            shard_plans=shard_plans,
            limit_per_file=args.limit_per_file,
        )
        write_json(shard_plan_path, [asdict(plan) for plan in shard_plans])
    else:
        # import-only 模式不重新扫描原始数据，只加载已有计划。
        shard_plans = load_shard_plans_from_file(shard_plan_path)

    import_results: list[dict] | None = None

    # import 阶段才会真正改数据库和 ES。
    # 它只消费 materialize 阶段留下的 shard_plan + JSONL 分片。
    if args.phase in {"all", "import"}:
        ensure_runtime()  # 确保 MySQL、ES、embedding 模型 运行正常
        kb = ensure_knowledge_base(args.kb_name)  # 确保知识库存在
        kb_folder = ensure_kb_folder(kb.name)  # 确保知识库虚拟目录存在
        shard_plans = load_shard_plans_from_file(shard_plan_path) # 加载分片计划
        import_results = import_shards(
            kb=kb,
            kb_folder=kb_folder,
            workdir=workdir,
            shard_plans=shard_plans,
            embed_batch_size=args.embed_batch_size,
            insert_batch_size=args.insert_batch_size,
        )
        write_json(workdir / "import_results.json", import_results)

    report = markdown_report(args.kb_name, analysis, shard_plans, import_results, workdir)
    write_text(report_path, report)
    logging.info("Done. Report written to %s", report_path)
