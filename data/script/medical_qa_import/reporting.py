"""Markdown 报告生成。"""

from __future__ import annotations

import json
from datetime import datetime

from .schema import SHARD_TARGET_COUNT, ShardPlan


def markdown_report(
    kb_name: str,
    analysis: dict | None,
    shard_plans: list[ShardPlan],
    import_results: list[dict] | None,
    workdir,
) -> str:
    """生成 Markdown 导入报告。

    报告分两种模式：

    1. materialize / all
       这时有扫描摘要，可以展示完整的数据统计。
    2. import-only
       这时通常不会重新扫描原始数据，只基于既有 shard_plan 和本次导入结果出报告。
    """

    lines: list[str] = []
    lines.append("# 医疗 QA 导入报告")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 目标知识库: `{kb_name}`")
    lines.append(f"- 工作目录: `{workdir}`")
    if analysis is not None:
        lines.append(f"- 数据目录: `{analysis['data_dir']}`")
    lines.append("")

    if analysis is not None:
        lines.append("## 1. 总体结果")
        lines.append("")
        lines.append(f"- 原始总条数: `{analysis['global_total_rows']}`")
        lines.append(f"- 空问题/空答案剔除数: `{analysis['global_invalid_rows']}`")
        lines.append(f"- 文件内重复命中数: `{analysis['global_local_duplicates']}`")
        lines.append(f"- 全局去重命中数: `{analysis['global_global_duplicates']}`")
        lines.append(f"- 全局保留条数: `{analysis['global_unique_rows']}`")
        lines.append(f"- 计划 logical documents 数: `{analysis['planned_shards']}`")
        lines.append("")

        lines.append("## 2. 分片规则")
        lines.append("")
        lines.append("- 一级按原始文件所属大类分组：儿科 / 内科 / 外科 / 妇产科 / 男科 / 肿瘤科。")
        lines.append("- 二级优先按 `department` 分桶；脏值或低频部门并入 `misc`。")
        lines.append(f"- 三级按每个 logical document 约 `{SHARD_TARGET_COUNT}` 条 QA 切片。")
        lines.append(f"- 默认独立分桶阈值: `{analysis['default_threshold']}`")
        lines.append(f"- 特殊阈值覆盖: `{json.dumps(analysis['threshold_overrides'], ensure_ascii=False)}`")
        lines.append("")

        lines.append("## 3. 文件扫描摘要")
        lines.append("")
        for item in analysis["file_stats"]:
            lines.append(f"### {item['filename']}")
            lines.append("")
            lines.append(f"- 一级大类: `{item['major_category']}`")
            lines.append(f"- 原始条数: `{item['total_rows']}`")
            lines.append(f"- 无效条数: `{item['invalid_rows']}`")
            lines.append(f"- 文件内重复: `{item['local_duplicate_rows']}`")
            lines.append(f"- 全局重复: `{item['global_duplicate_rows']}`")
            lines.append(f"- 最终保留: `{item['kept_rows']}`")
            lines.append(f"- 干净 department 条数: `{item['clean_department_rows']}`")
            lines.append(f"- 脏 department 条数: `{item['dirty_department_rows']}`")
            lines.append(f"- 去重后 department 去重数: `{item['unique_departments']}`")
            if item["top_departments"]:
                top_text = "，".join(f"{name}({count})" for name, count in item["top_departments"][:10])
                lines.append(f"- Top departments: {top_text}")
            if item["dirty_department_samples"]:
                lines.append("- 脏 department 示例:")
                for sample in item["dirty_department_samples"][:6]:
                    lines.append(f"  - {sample}")
            lines.append("")
    else:
        lines.append("## 1. 运行模式说明")
        lines.append("")
        lines.append("- 本次报告来自 `import-only` 运行。")
        lines.append("- 脚本没有重新扫描原始 QA 目录，因此不包含源数据统计摘要。")
        lines.append("- 当前报告主要记录分片计划和本次导入执行结果。")
        lines.append("")

    lines.append("## 4. 逻辑文档计划")
    lines.append("")
    for plan in shard_plans:
        lines.append(
            f"- `{plan.shard_name}`: `{plan.major_category}` / `{plan.bucket_department}` / "
            f"`{plan.bucket_type}` / 计划 `{plan.planned_count}` 条"
        )
    lines.append("")

    if import_results is not None:
        lines.append("## 5. 导入执行结果")
        lines.append("")
        imported = [item for item in import_results if item["status"] == "imported"]
        skipped = [item for item in import_results if item["status"] == "skipped_existing"]
        lines.append(f"- 新导入分片数: `{len(imported)}`")
        lines.append(f"- 复用已有分片数: `{len(skipped)}`")
        lines.append(f"- 新写入 chunk 总数: `{sum(item['chunk_count'] for item in imported)}`")
        lines.append(f"- 新写入 token 总数: `{sum(item['token_num'] for item in imported)}`")
        lines.append("")
        for item in import_results:
            lines.append(
                f"- `{item['shard_name']}` / `{item['doc_id']}` / `{item['status']}` / "
                f"chunks=`{item['chunk_count']}` / tokens=`{item['token_num']}`"
            )
        lines.append("")

    lines.append("## 6. 输出产物")
    lines.append("")
    lines.append("- `shard_plan.json`")
    lines.append("- `normalized_shards/*.jsonl`")
    if import_results is not None:
        lines.append("- `import_results.json`")
    lines.append("- `reports/qa_import_report.md`")
    lines.append("")
    return "\n".join(lines)
