"""医疗 PDF 建库脚本的命令行入口。

这个模块只做“流程编排”，不承载具体业务细节。
这样以后如果要做：
- GUI 包装
- Notebook 调用
- 远程任务调度

都可以复用这里的流程，而不需要重新拼装 planning / importing。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from .importing import ensure_kb_folder, ensure_knowledge_base, ensure_runtime, import_shards
from .io_utils import load_shard_plans_from_file, write_json, write_text
from .mineru import MinerURunner
from .planning import materialize_shards, scan_corpus
from .reporting import markdown_report
from .schema import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TOKEN_NUM,
    DEFAULT_DATA_DIR,
    DEFAULT_HEADING_SPLIT_DEPTH,
    DEFAULT_LOGICAL_DOC_MAX_CHUNKS,
    DEFAULT_LOGICAL_DOC_TOKEN_NUM,
    DEFAULT_MINERU_PARSE_METHOD,
    DEFAULT_WORKDIR,
    PHASES,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    参数设计沿用 QA 脚本的整体思路，但针对 PDF 补了两类特有控制项：
    1. MinerU 调用方式和解析模式
    2. 章节切分 / logical document 切分阈值
    """

    parser = argparse.ArgumentParser(description="Import medical PDFs into a MinerU-based knowledge base.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="PDF directory.")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR), help="Output and checkpoint directory.")
    parser.add_argument("--kb-name", default="medical_pdf_kb", help="Knowledge base name.")
    # `materialize` 与 `import` 解耦，是整套脚本最重要的设计之一。
    parser.add_argument("--phase", default="all", choices=sorted(PHASES), help="Run materialize/import/all.")
    parser.add_argument("--limit-files", type=int, default=None, help="Only process the first N PDF files.")
    parser.add_argument("--mineru-mode", default="auto", choices=["auto", "api", "cli"], help="How to call MinerU.")
    parser.add_argument("--mineru-api-url", default="", help="MinerU API base URL.")
    parser.add_argument("--mineru-server-url", default="", help="Alternative MinerU server URL.")
    parser.add_argument("--mineru-output-dir", default="", help="Fixed CLI output directory for MinerU.")
    parser.add_argument(
        "--mineru-parse-method",
        default=DEFAULT_MINERU_PARSE_METHOD,
        choices=["auto", "ocr", "text", "txt"],
        help="MinerU parse method.",
    )
    # 强制重跑 MinerU 主要用于：
    # - 改了 parse_method
    # - 调整了 markdown 清洗规则
    # - 怀疑旧缓存质量不好
    parser.add_argument("--force-reparse", action="store_true", help="Ignore cached markdown / blocks and rerun MinerU.")
    parser.add_argument("--chunk-token-num", type=int, default=DEFAULT_CHUNK_TOKEN_NUM, help="Target token count for each chunk.")
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_TOKENS,
        help="Optional overlap budget between consecutive chunks.",
    )
    parser.add_argument(
        "--logical-doc-token-num",
        type=int,
        default=DEFAULT_LOGICAL_DOC_TOKEN_NUM,
        help="Target token budget for each logical document.",
    )
    parser.add_argument(
        "--logical-doc-max-chunks",
        type=int,
        default=DEFAULT_LOGICAL_DOC_MAX_CHUNKS,
        help="Maximum chunks in each logical document.",
    )
    parser.add_argument(
        "--heading-split-depth",
        type=int,
        default=DEFAULT_HEADING_SPLIT_DEPTH,
        help="How many heading levels are used to define chapter roots.",
    )
    parser.add_argument("--embed-batch-size", type=int, default=128, help="Embedding batch size.")
    parser.add_argument("--insert-batch-size", type=int, default=64, help="ES insert batch size.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    """脚本总入口。

    - 若包含 materialize，就先生成中间产物
    - 若包含 import，就消费中间产物入库
    - 最后始终输出一份 Markdown 报告
    """

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # 统一把路径提前 resolve，后面写报告和生成相对路径时更稳定。
    data_dir = Path(args.data_dir).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    pdf_manifest_path = workdir / "pdf_manifest.json"
    shard_plan_path = workdir / "shard_plan.json"
    report_path = workdir / "reports" / "pdf_import_report.md"

    analysis: dict | None = None
    shard_plans = []

    if args.phase in {"all", "materialize"}:
        # 先做源 PDF 扫描，并把 manifest 固化下来。
        _, pdf_sources = scan_corpus(data_dir, args.limit_files)
        write_json(pdf_manifest_path, [asdict(source) for source in pdf_sources])

        # 这里仅构造一个统一的 MinerU 运行器，
        # materialize 内部不再关心底层到底走 API 还是 CLI。
        runner = MinerURunner(
            mode=args.mineru_mode,
            api_url=args.mineru_api_url,
            server_url=args.mineru_server_url,
            output_dir=args.mineru_output_dir,
            delete_output=not bool(args.mineru_output_dir),
        )
        analysis, shard_plans = materialize_shards(
            workdir=workdir,
            pdf_sources=pdf_sources,
            runner=runner,
            parse_method=args.mineru_parse_method,
            chunk_token_num=args.chunk_token_num,
            chunk_overlap_tokens=args.chunk_overlap_tokens,
            logical_doc_token_num=args.logical_doc_token_num,
            logical_doc_max_chunks=args.logical_doc_max_chunks,
            heading_split_depth=args.heading_split_depth,
            force_reparse=args.force_reparse,
        )
        write_json(shard_plan_path, [asdict(plan) for plan in shard_plans])
    else:
        # import-only 模式不再重扫原始 PDF，而是直接复用已有 shard_plan。
        shard_plans = load_shard_plans_from_file(shard_plan_path)

    import_results: list[dict] | None = None
    if args.phase in {"all", "import"}:
        # import 阶段才真正访问 MySQL / ES / embedding。
        ensure_runtime()  # 确保运行时环境已初始化
        kb = ensure_knowledge_base(args.kb_name)  # 确保知识库存在
        kb_folder = ensure_kb_folder(kb.name)  # 确保知识库文件夹存在
        shard_plans = load_shard_plans_from_file(shard_plan_path)  # 确保 shard_plan 存在
        import_results = import_shards(
            kb=kb,
            kb_folder=kb_folder,
            workdir=workdir,
            shard_plans=shard_plans,
            embed_batch_size=args.embed_batch_size,
            insert_batch_size=args.insert_batch_size,
        )
        write_json(workdir / "import_results.json", import_results)

    # 无论跑到哪一步，最后都输出报告，方便留痕和排障。
    report = markdown_report(args.kb_name, analysis, shard_plans, import_results, workdir)
    write_text(report_path, report)
    logging.info("Done. Report written to %s", report_path)
