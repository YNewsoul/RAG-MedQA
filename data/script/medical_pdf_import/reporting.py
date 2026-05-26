"""导入报告生成。

这份报告的目标不是“漂亮”，而是“排障和复盘友好”：
- 本轮处理了多少 PDF
- 每本 PDF 切出了多少 blocks/chunks/shards
- 哪些分片是新导入，哪些是复用
"""

from __future__ import annotations

from pathlib import Path

from .schema import ShardPlan


def markdown_report(
    kb_name: str,
    analysis: dict | None,
    shard_plans: list[ShardPlan],
    import_results: list[dict] | None,
    workdir: Path,
) -> str:
    """生成 Markdown 报告。

    报告被分成 4 段：
    1. 任务概况
    2. 源 PDF 摘要
    3. 分片计划
    4. 入库结果

    这样既方便人工检查，也方便后续作为实验记录保留下来。
    """

    lines: list[str] = []
    lines.append("# 医疗 PDF 知识库导入报告")
    lines.append("")
    lines.append("## 1. 任务概况")
    lines.append("")
    lines.append(f"- 知识库名称：`{kb_name}`")
    lines.append(f"- 工作目录：`{workdir}`")
    lines.append(f"- logical documents 数：`{len(shard_plans)}`")

    if analysis:
        # 这一段只在本轮跑过 materialize 时才有完整统计。
        lines.append(f"- 源 PDF 数：`{analysis.get('source_count', 0)}`")
        lines.append(f"- 总页数：`{analysis.get('total_pages', 0)}`")
        lines.append(f"- 总大小（字节）：`{analysis.get('total_size', 0)}`")
        lines.append(f"- MinerU 复用数：`{analysis.get('reused_parse_count', 0)}`")
        lines.append(f"- MinerU 新解析数：`{analysis.get('fresh_parse_count', 0)}`")
        lines.append(f"- 解析块总数：`{analysis.get('total_blocks', 0)}`")
        lines.append(f"- 表格块总数：`{analysis.get('total_tables', 0)}`")
        lines.append(f"- chunk 总数：`{analysis.get('total_chunks', 0)}`")

    lines.append("")
    lines.append("## 2. 源 PDF 摘要")
    lines.append("")
    if analysis and analysis.get("sources"):
        for item in analysis["sources"]:
            lines.append(
                "- `{}` | pages=`{}` | blocks=`{}` | tables=`{}` | chunks=`{}` | shards=`{}` | reused=`{}`".format(
                    item["source_name"],
                    item.get("page_count", 0),
                    item.get("block_count", 0),
                    item.get("table_count", 0),
                    item.get("chunk_count", 0),
                    item.get("shard_count", 0),
                    item.get("reused_parse", False),
                )
            )
    else:
        lines.append("- 本轮未执行 materialize，暂无源 PDF 统计。")

    lines.append("")
    lines.append("## 3. 分片计划")
    lines.append("")
    if shard_plans:
        # 这里展示的是 logical document 级别的计划，而不是原始 PDF 级别。
        for plan in shard_plans:
            lines.append(
                "- `{}` | source=`{}` | chapter=`{}` | chunks=`{}` | tokens≈`{}`".format(
                    plan.shard_name,
                    plan.source_name,
                    plan.chapter_root,
                    plan.planned_count,
                    plan.token_estimate,
                )
            )
    else:
        lines.append("- 当前没有可导入的 logical document。")

    lines.append("")
    lines.append("## 4. 入库结果")
    lines.append("")
    if import_results is None:
        lines.append("- 本轮未执行 import。")
    else:
        # `imported` 与 `skipped_existing` 分开统计，方便判断续跑复用效果。
        imported = [item for item in import_results if item["status"] == "imported"]
        skipped = [item for item in import_results if item["status"] == "skipped_existing"]
        lines.append(f"- 新导入分片数：`{len(imported)}`")
        lines.append(f"- 复用旧分片数：`{len(skipped)}`")
        lines.append(f"- 新写入 token 总数：`{sum(item['token_num'] for item in imported)}`")
        for item in import_results:
            lines.append(
                "- `{}` | status=`{}` | chunks=`{}` | tokens=`{}`".format(
                    item["shard_name"],
                    item["status"],
                    item["chunk_count"],
                    item["token_num"],
                )
            )

    return "\n".join(lines) + "\n"
