"""分片规划与 JSONL 物化。

这个模块负责把原始 QA 目录转换成标准化中间产物：
- `shard_plan.json`
- `normalized_shards/*.jsonl`

它不接触数据库，也不写 ES，职责保持纯净。
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .io_utils import load_json_records
from .schema import (
    MAJOR_ORDER,
    SHARD_TARGET_COUNT,
    THRESHOLD_OVERRIDES,
    FileStats,
    ShardPlan,
    dedupe_hash,
    department_is_clean,
    major_from_filename,
    normalize_text,
    shard_rel_path,
    threshold_for_major,
)


def scan_corpus(
    data_dir: Path,
    limit_per_file: int | None,
    default_threshold: int,
) -> tuple[dict, list[ShardPlan]]:
    """扫描整个 QA 目录并生成分片计划

    它会完成：
    - 逐文件统计
    - 空值过滤
    - 全局去重
    - department 清洗
    - bucket 规模计算
    - shard 计划生成
    """

    global_seen: set[str] = set()
    file_stats: list[FileStats] = []
    clean_department_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dirty_department_counts: Counter[str] = Counter()
    bucket_counts: Counter[tuple[str, str]] = Counter()
    source_files_by_major: dict[str, str] = {}
    dirty_department_examples: dict[str, list[str]] = defaultdict(list)

    ordered_files = sorted(data_dir.glob("*.json"))
    if not ordered_files:
        raise FileNotFoundError(f"No JSON files found under {data_dir}")

    # 第一轮：逐文件扫描，得到每个一级大类下干净科室和脏值的统计。
    for path in ordered_files:
        major = major_from_filename(path.name)
        source_files_by_major[major] = path.name
        rows = load_json_records(path, limit=limit_per_file)

        local_seen: set[str] = set()
        department_counter: Counter[str] = Counter()

        stats = FileStats(
            filename=path.name,
            major_category=major,
            size_bytes=path.stat().st_size,
            dirty_department_samples=[],
        )

        for item in rows:
            stats.total_rows += 1

            # 四个核心字段统一规范化，避免空白差异影响统计。
            title = normalize_text(item.get("title"))
            ask = normalize_text(item.get("ask"))
            answer = normalize_text(item.get("answer"))
            department = normalize_text(item.get("department"))

            # 没有问题或没有答案的条目，没有导入价值，直接丢弃。
            if not ask or not answer:
                stats.invalid_rows += 1
                continue

            qa_key = dedupe_hash(ask, answer)

            # 文件内重复只做统计，不立即丢弃；真正是否保留由全局去重决定。
            if qa_key in local_seen:
                stats.local_duplicate_rows += 1
            else:
                local_seen.add(qa_key)

            # 全局只保留第一次出现的问答。
            if qa_key in global_seen:
                stats.global_duplicate_rows += 1
                continue
            global_seen.add(qa_key)

            stats.kept_rows += 1

            if department_is_clean(department):
                stats.clean_department_rows += 1
                department_counter[department] += 1
                clean_department_counts[major][department] += 1
            else:
                stats.dirty_department_rows += 1
                dirty_department_counts[major] += 1
                if department and len(dirty_department_examples[major]) < 12:
                    dirty_department_examples[major].append(department[:180])

            # title 当前不参与分桶，但后续构造 chunk 时会进入问题文本和展示文本。
            _ = title

        stats.unique_departments = len(department_counter)
        stats.top_departments = department_counter.most_common(20)
        stats.dirty_department_samples = dirty_department_examples[major]
        file_stats.append(stats)

        # 基于累计统计判断哪些科室值得独立成桶。
        separate_departments = {
            department
            for department, count in clean_department_counts[major].items()
            if count >= threshold_for_major(major, default_threshold)
        }

        # 这里每轮都重建一次 major 级 bucket 统计，而不是在旧结果上继续叠加。
        # 因为随着累计计数变化，“某个 department 是否达到独立成桶阈值”也会变化。
        for key in [key for key in list(bucket_counts.keys()) if key[0] == major]:
            del bucket_counts[key]

        for department, count in clean_department_counts[major].items():
            bucket_department = department if department in separate_departments else "misc"
            bucket_counts[(major, bucket_department)] += count

        if dirty_department_counts[major]:
            bucket_counts[(major, "misc")] += dirty_department_counts[major]

    # 第二轮：把每个 bucket 再切成若干 8000 条左右的 logical documents。
    shard_plans: list[ShardPlan] = []
    for major in MAJOR_ORDER:
        if major not in source_files_by_major:
            continue

        source_file = source_files_by_major[major]
        buckets = [
            (bucket_department, count)
            for (bucket_major, bucket_department), count in bucket_counts.items()
            if bucket_major == major and count > 0
        ]
        buckets.sort(key=lambda item: (item[0] == "misc", -item[1], item[0]))

        for bucket_department, count in buckets:
            parts = math.ceil(count / SHARD_TARGET_COUNT)
            remaining = count
            for part_index in range(1, parts + 1):
                planned_count = min(SHARD_TARGET_COUNT, remaining)
                remaining -= planned_count
                shard_name = f"qa_{major}_{bucket_department}_p{part_index:03d}"
                shard_plans.append(
                    ShardPlan(
                        major_category=major,
                        source_file=source_file,
                        bucket_department=bucket_department,
                        bucket_type="clean_department" if bucket_department != "misc" else "long_tail",
                        part_index=part_index,
                        shard_name=shard_name,
                        planned_count=planned_count,
                        shard_file=shard_rel_path(shard_name),
                    )
                )

    analysis = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "limit_per_file": limit_per_file,
        "global_unique_rows": len(global_seen),
        "global_total_rows": sum(item.total_rows for item in file_stats),
        "global_invalid_rows": sum(item.invalid_rows for item in file_stats),
        "global_local_duplicates": sum(item.local_duplicate_rows for item in file_stats),
        "global_global_duplicates": sum(item.global_duplicate_rows for item in file_stats),
        "default_threshold": default_threshold,
        "threshold_overrides": THRESHOLD_OVERRIDES,
        "chunk_target": SHARD_TARGET_COUNT,
        "file_stats": [asdict(item) for item in file_stats],
        "bucket_summary": {
            major: [
                {
                    "bucket_department": bucket_department,
                    "count": count,
                }
                for (bucket_major, bucket_department), count in sorted(
                    bucket_counts.items(),
                    key=lambda item: (
                        MAJOR_ORDER.index(item[0][0]),
                        item[0][1] == "misc",
                        -item[1],
                        item[0][1],
                    ),
                )
                if bucket_major == major
            ]
            for major in MAJOR_ORDER
            if major in source_files_by_major
        },
        "planned_shards": len(shard_plans),
    }
    return analysis, shard_plans


def materialize_shards(
    data_dir: Path,
    workdir: Path,
    shard_plans: list[ShardPlan],
    limit_per_file: int | None,
) -> list[ShardPlan]:
    """把分片计划真正展开成很多 JSONL 文件。

    这一步不会改数据库，但它是整个脚本最关键的中间层：
    - 每个 JSONL 就是一个 future document
    - import 阶段只关心这些 JSONL
    - 一旦中途失败，可以复用这批标准化中间产物

    换句话说：
    - materialize 解决“原始数据怎么切、怎么清洗、怎么定稿”
    - import 解决“已经定稿的分片怎么稳定写进系统”
    """

    shard_dir = workdir / "normalized_shards"
    if shard_dir.exists():
        # 物化阶段始终重建，确保分片文件和当前计划完全一致。
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    plan_lookup = {
        (plan.major_category, plan.bucket_department, plan.part_index): plan
        for plan in shard_plans
    }
    separate_departments = {
        major: {
            plan.bucket_department
            for plan in shard_plans
            if plan.major_category == major and plan.bucket_department != "misc"
        }
        for major in MAJOR_ORDER
    }

    bucket_offsets: Counter[tuple[str, str]] = Counter()
    global_seen: set[str] = set()
    open_handles: dict[str, tuple[object, hashlib._Hash]] = {}
    written_counts: Counter[str] = Counter()

    try:
        for path in sorted(data_dir.glob("*.json")):
            major = major_from_filename(path.name)
            rows = load_json_records(path, limit=limit_per_file)
            for row_index, item in enumerate(rows):
                title = normalize_text(item.get("title"))
                ask = normalize_text(item.get("ask"))
                answer = normalize_text(item.get("answer"))
                department = normalize_text(item.get("department"))

                if not ask or not answer:
                    continue

                qa_key = dedupe_hash(ask, answer)
                if qa_key in global_seen:
                    continue
                global_seen.add(qa_key)

                # 只有足够干净且达到独立成桶标准的 department 才会保留原值；
                # 其余全部并入 misc。
                clean_department = department if department_is_clean(department) else ""
                bucket_department = clean_department if clean_department in separate_departments[major] else "misc"
                bucket_key = (major, bucket_department)

                # 同一 bucket 内，按写入顺序计算它应该落到第几个分片。
                bucket_offsets[bucket_key] += 1
                part_index = ((bucket_offsets[bucket_key] - 1) // SHARD_TARGET_COUNT) + 1
                plan = plan_lookup[(major, bucket_department, part_index)]
                shard_path = workdir / plan.shard_file

                # question_text 偏检索，content_with_weight 偏向量与展示。
                question_text = f"{title} {ask}".strip() if title else ask
                content_with_weight = (
                    f"标题：{title}\n问题：{ask}\n回答：{answer}"
                    if title
                    else f"问题：{ask}\n回答：{answer}"
                )
                normalized = {
                    "title": title,
                    "ask": ask,
                    "answer": answer,
                    "question_text": question_text,
                    "content_with_weight": content_with_weight,
                    "major_category": major,
                    "raw_department": department,
                    "clean_department": clean_department,
                    "bucket_department": bucket_department,
                    "source_file": path.name,
                    "source_row_index": row_index,
                }

                if str(shard_path) not in open_handles:
                    shard_path.parent.mkdir(parents=True, exist_ok=True)
                    open_handles[str(shard_path)] = (
                        shard_path.open("w", encoding="utf-8"),
                        hashlib.md5(),
                    )

                handle, md5_obj = open_handles[str(shard_path)]
                line = json.dumps(normalized, ensure_ascii=False) + "\n"
                handle.write(line)
                md5_obj.update(line.encode("utf-8"))
                written_counts[plan.shard_name] += 1
    finally:
        for handle, _ in open_handles.values():
            handle.close()

    # 物化完成后，把文件哈希和大小回填到 plan 中，供 import 阶段做幂等校验。
    updated_plans: list[ShardPlan] = []
    for plan in shard_plans:
        shard_path = workdir / plan.shard_file
        actual_count = written_counts[plan.shard_name]
        if actual_count != plan.planned_count:
            raise RuntimeError(
                f"Shard {plan.shard_name} planned {plan.planned_count} rows but materialized {actual_count} rows."
            )

        data = shard_path.read_bytes()
        updated_plans.append(
            ShardPlan(
                major_category=plan.major_category,
                source_file=plan.source_file,
                bucket_department=plan.bucket_department,
                bucket_type=plan.bucket_type,
                part_index=plan.part_index,
                shard_name=plan.shard_name,
                planned_count=plan.planned_count,
                shard_file=plan.shard_file,
                doc_id=plan.doc_id,
                file_md5=hashlib.md5(data).hexdigest(),
                file_size=len(data),
            )
        )
    return updated_plans
