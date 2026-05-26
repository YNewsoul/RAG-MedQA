"""PDF 语料扫描、MinerU 预处理与分片物化。

这个模块承担整个脚本最“重”的前半段工作：

1. 扫描原始 PDF
2. 跑 MinerU，把 PDF 结构化成 markdown
3. 把 markdown 继续解析成脚本自己的块结构
4. 把块结构切成 chunk
5. 再把 chunk 组织成 logical document JSONL

最后产出的 `normalized_shards/*.jsonl`，
就是 import 阶段真正消费的标准中间层。
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from collections import Counter
from dataclasses import asdict
from html import unescape
from pathlib import Path

import pdfplumber

from .io_utils import load_mineru_cache, read_json, write_json, write_jsonl, write_text
from .mineru import MinerURunner
from .schema import (
    DEFAULT_HEADING_SPLIT_DEPTH,
    DEFAULT_LOGICAL_DOC_MAX_CHUNKS,
    DEFAULT_LOGICAL_DOC_TOKEN_NUM,
    PdfSourceInfo,
    MinerUCacheInfo,
    ShardPlan,
    deterministic_source_id,
    file_md5,
    heading_path_to_text,
    infer_pdf_title,
    infer_specialty,
    make_shard_name,
    normalize_text,
)
from common.token_utils import num_tokens_from_string


def count_pdf_pages(path: Path) -> int:
    """统计 PDF 页数；失败时返回 0。"""

    # 页数不是强依赖字段，所以失败时不要中断整个流程。
    # 返回 0 并在日志里标记即可。
    try:
        with pdfplumber.open(str(path)) as pdf:
            return len(pdf.pages)
    except Exception:  # noqa: BLE001
        logging.exception("Failed to count PDF pages: %s", path)
        return 0


def scan_corpus(data_dir: Path, limit_files: int | None = None) -> tuple[dict, list[PdfSourceInfo]]:
    """扫描原始 PDF 目录，生成基础 manifest。

    这一步只做“源文件事实采集”，不做任何 MinerU 或切块操作。
    它的目标是先把这批 PDF 的规模和身份固定下来。
    """

    pdf_paths = sorted(
        [path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"],
        key=lambda item: item.name,
    )
    if limit_files is not None:
        pdf_paths = pdf_paths[:limit_files]

    sources: list[PdfSourceInfo] = []
    total_size = 0
    total_pages = 0
    for path in pdf_paths:
        # 这里把 md5、页数、标题、专题名都提前算出来，
        # 方便后面的缓存复用、报告生成和 logical document 命名。
        source_md5 = file_md5(path)
        file_size = path.stat().st_size
        page_count = count_pdf_pages(path)
        source = PdfSourceInfo(
            source_name=path.name,
            source_path=str(path),
            source_md5=source_md5,
            file_size=file_size,
            page_count=page_count,
            title=infer_pdf_title(path.name),
            specialty=infer_specialty(path.name),
            source_id=deterministic_source_id(path.name),
        )
        sources.append(source)
        total_size += file_size
        total_pages += page_count

    analysis = {
        "source_count": len(sources),
        "total_size": total_size,
        "total_pages": total_pages,
        "sources": [asdict(source) for source in sources],
    }
    return analysis, sources


def table_markdown_to_text(table_markdown: str) -> str:
    """把 markdown table 转成更适合 embedding / 搜索的纯文本。

    这样做的原因是：
    - Markdown 表格原样做 embedding，噪声通常较大
    - 把单元格规整成“行文本”，更利于语义检索
    """

    rows: list[str] = []
    for line in table_markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        # 跳过 markdown table 的分隔线，例如 `| --- | --- |`
        if all(re.fullmatch(r":?-+:?", cell or "-") for cell in cells):
            continue
        normalized_cells = [normalize_text(cell) for cell in cells if normalize_text(cell)]
        if normalized_cells:
            rows.append(" | ".join(normalized_cells))
    return "\n".join(rows).strip()


def html_fragment_to_text(html_fragment: str) -> str:
    """把 HTML 片段压平成适合检索的普通文本。"""

    text = re.sub(r"(?is)<br\s*/?>", "\n", html_fragment)
    text = re.sub(r"(?is)</p\s*>", "\n", text)
    text = re.sub(r"(?is)</div\s*>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return normalize_text(unescape(text))


def table_html_to_text(table_html: str) -> str:
    """把 HTML table 转成行文本。

    MinerU 在部分 PDF 上不会返回 markdown 管道表，而是直接返回
    `<table><tr><td>...</td>...</tr></table>`。如果这一层不处理，后续会把表格整段
    当普通文本吞掉，既丢结构，也会让 `table_count` 失真。
    """

    row_texts: list[str] = []
    for row_html in re.findall(r"(?is)<tr\b[^>]*>(.*?)</tr>", table_html):
        cells = re.findall(r"(?is)<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html)
        normalized_cells = [html_fragment_to_text(cell) for cell in cells]
        normalized_cells = [cell for cell in normalized_cells if cell]
        if normalized_cells:
            row_texts.append(" | ".join(normalized_cells))

    if row_texts:
        return "\n".join(row_texts).strip()

    # 兜底：即使表格结构很怪，也尽量留下纯文本，不让整块内容丢失。
    return html_fragment_to_text(table_html)


def parse_markdown_to_blocks(markdown_text: str) -> list[dict]:
    """把 MinerU 输出的 markdown 转成块列表。

    输出块是脚本自己的中间结构，不依赖项目原有 PDF parser。
    """

    # 空 markdown 直接返回空块列表。
    if not markdown_text or not markdown_text.strip():
        return []

    blocks: list[dict] = []
    heading_stack: list[dict] = []
    current_para: list[str] = []
    current_table: list[str] = []
    current_table_format = ""

    def current_heading_path() -> list[str]:
        # heading_stack 里保存的是当前仍然生效的标题栈，
        # 这里把它转成纯文本路径，供 paragraph/table 继承。
        return [item["text"] for item in heading_stack]

    def flush_paragraph() -> None:
        # paragraph 的定义很保守：由空行分隔的一段普通文本。
        if not current_para:
            return
        text = normalize_text("\n".join(current_para))
        current_para.clear()
        if not text:
            return
        blocks.append(
            {
                "type": "paragraph",
                "text": text,
                "heading_path": current_heading_path(),
            }
        )

    def flush_table() -> None:
        nonlocal current_table_format
        if not current_table:
            current_table_format = ""
            return
        table_format = current_table_format
        raw = "\n".join(current_table).strip()
        # 表格同时保留：
        # 1. 原始 markdown（便于排障）
        # 2. 转换后的纯文本（便于后续检索）
        if table_format == "html":
            text = table_html_to_text(raw)
        else:
            text = table_markdown_to_text(raw)
        current_table.clear()
        current_table_format = ""
        if not text:
            return
        block = {
            "type": "table",
            "text": text,
            "heading_path": current_heading_path(),
        }
        if table_format == "html":
            block["raw_html"] = raw
        else:
            block["raw_markdown"] = raw
        blocks.append(block)

    def is_table_row(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.count("|") >= 2

    for line in markdown_text.splitlines():
        stripped = line.strip()

        # MinerU 常会插入裸图片引用，这一版先跳过，避免污染文本块。
        if re.match(r"^!\[.*?\]\(.*?\)\s*$", stripped):
            continue

        # 多行 HTML table 模式：进入之后持续累积，直到读到 `</table>`。
        if current_table_format == "html":
            current_table.append(line)
            if "</table>" in stripped.lower():
                flush_table()
            continue

        # 单行 / 起始行 HTML table。MinerU 在不少 PDF 上会直接返回这种结构。
        if "<table" in stripped.lower():
            flush_paragraph()
            current_table_format = "html"
            current_table.append(line)
            if "</table>" in stripped.lower():
                flush_table()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading_match:
            # 标题出现时，先把之前积累的正文/表格都落盘，
            # 然后更新 heading stack。
            flush_paragraph()
            flush_table()
            level = len(heading_match.group(1))
            title = normalize_text(heading_match.group(2))
            # 标题层级回退时，弹出比当前层级更深或相同层级的旧标题。
            while heading_stack and heading_stack[-1]["level"] >= level:
                heading_stack.pop()
            heading_stack.append({"level": level, "text": title})
            blocks.append(
                {
                    "type": "heading",
                    "text": title,
                    "heading_level": level,
                    "heading_path": current_heading_path(),
                }
            )
            continue

        if is_table_row(line):
            # 表格与普通段落分开累计，因为后面会走不同的处理逻辑。
            if current_para:
                flush_paragraph()
            current_table_format = "markdown"
            current_table.append(line)
            continue

        if current_table_format == "markdown":
            flush_table()

        if not stripped:
            flush_paragraph()
            continue

        current_para.append(line)

    # 文件末尾的残留正文/表格也要记得落盘。
    flush_paragraph()
    flush_table()
    return filter_noise_blocks(blocks)


def filter_noise_blocks(blocks: list[dict]) -> list[dict]:
    """做一层轻量噪声清理。

    这里不追求复杂规则，只移除最常见的页码/页眉页脚型噪声。
    """

    # 统计短段落重复次数，后面用于过滤高频页眉页脚噪声。
    short_para_counter = Counter(
        block["text"]
        for block in blocks
        if block["type"] == "paragraph" and len(block["text"]) <= 30
    )

    filtered: list[dict] = []
    for block in blocks:
        text = block["text"]
        if block["type"] == "paragraph":
            # 先过滤显而易见的页码样式。
            if re.fullmatch(r"第?\s*\d+\s*页", text):
                continue
            if re.fullmatch(r"\d+", text):
                continue
            # 再过滤高频重复短文本，例如很多页都重复的页眉页脚。
            if short_para_counter[text] >= 4:
                continue
        filtered.append(block)
    return filtered


def source_cache_paths(workdir: Path, source: PdfSourceInfo) -> tuple[Path, Path]:
    """返回单个 PDF 的 markdown / blocks 缓存文件路径。"""

    # 这里统一用 source_id 命名，而不是直接拿中文文件名做缓存文件名，
    # 目的是避免路径编码、特殊字符和过长文件名问题。
    markdown_path = workdir / "mineru_markdown" / f"{source.source_id}.md"
    blocks_path = workdir / "parsed_blocks" / f"{source.source_id}.json"
    return markdown_path, blocks_path


def load_or_parse_blocks(
    source: PdfSourceInfo,
    runner: MinerURunner,
    workdir: Path,
    parse_method: str,
    cache_index: dict[str, MinerUCacheInfo],
    force_reparse: bool = False,
) -> tuple[list[dict], bool, MinerUCacheInfo]:
    """读取或生成单个 PDF 的 markdown / blocks 缓存。

    返回值里的 `bool` 表示这次是否复用了旧缓存。
    上层会把它计入报告，用来评估 materialize 的复用效果。
    """

    markdown_path, blocks_path = source_cache_paths(workdir, source)
    cached = cache_index.get(source.source_name)

    # 只有在“源文件没变 + 两个缓存文件都还在”时，才认为可复用。
    if (
        not force_reparse
        and cached is not None
        and cached.source_md5 == source.source_md5
        and markdown_path.exists()
        and blocks_path.exists()
    ):
        blocks = read_json(blocks_path, default=[]) or []
        return blocks, True, cached

    # 到这里说明不能复用，必须重新调用 MinerU。
    markdown_text = runner.parse_to_markdown(Path(source.source_path), parse_method=parse_method)
    blocks = parse_markdown_to_blocks(markdown_text)
    table_count = sum(1 for block in blocks if block["type"] == "table")

    # markdown 和解析后的 blocks 都保留下来：
    # markdown 便于人工检查，blocks 便于后续快速 materialize。
    write_text(markdown_path, markdown_text)
    write_json(blocks_path, blocks)

    cache_info = MinerUCacheInfo(
        source_name=source.source_name,
        source_md5=source.source_md5,
        markdown_file=str(markdown_path.relative_to(workdir)).replace("\\", "/"),
        block_file=str(blocks_path.relative_to(workdir)).replace("\\", "/"),
        block_count=len(blocks),
        table_count=table_count,
    )
    return blocks, False, cache_info


def blocks_to_units(
    source: PdfSourceInfo,
    blocks: list[dict],
    heading_split_depth: int,
) -> list[dict]:
    """把解析块转成后续 chunk 合并使用的内容单元。

    `unit` 是介于“原始块”和“最终 chunk”之间的一层抽象：
    - 比 block 更贴近业务语义
    - 比 chunk 更细
    它让后续 chunk 合并逻辑更清晰。
    """

    units: list[dict] = []
    for block in blocks:
        # 标题块只用于确定路径，不直接变成正文单元。
        if block["type"] == "heading":
            continue

        heading_path = [normalize_text(item) for item in block.get("heading_path", []) if normalize_text(item)]
        section_path = heading_path_to_text(heading_path) or source.title
        chapter_root = heading_path_to_text(heading_path[:heading_split_depth]) or source.specialty or source.title
        body_text = normalize_text(block["text"])
        if not body_text:
            continue
        # 给表格显式加前缀，帮助 embedding 模型理解这是结构化内容而不是自然段。
        rendered_text = f"表格内容：\n{body_text}" if block["type"] == "table" else body_text
        token_estimate = max(1, num_tokens_from_string(f"{section_path}\n{rendered_text}"))
        units.append(
            {
                "block_type": block["type"],
                "text": rendered_text,
                "section_path": section_path,
                "heading_path": heading_path,
                "chapter_root": chapter_root,
                "token_estimate": token_estimate,
                "table_count": 1 if block["type"] == "table" else 0,
            }
        )
    return units


def tail_units_for_overlap(units: list[dict], overlap_tokens: int) -> list[dict]:
    """按 token 预算取上一 chunk 尾部若干单元，作为下一 chunk 的上下文。"""

    # overlap 预算为 0 时，直接不做重叠。
    if overlap_tokens <= 0:
        return []
    carried: list[dict] = []
    total = 0
    for unit in reversed(units):
        copied = dict(unit)
        # 这里保留一个显式标记，方便以后如果要调试 chunk 来源能看出哪些是重叠带进来的。
        copied["is_overlap"] = True
        carried.insert(0, copied)
        total += copied["token_estimate"]
        if total >= overlap_tokens:
            break
    return carried


def finalize_chunk_row(
    source: PdfSourceInfo,
    chapter_root: str,
    units: list[dict],
) -> dict:
    """把一组内容单元收束成最终 chunk 记录。

    这一步会把来源文档、专题、章节路径和正文合成为最终的
    `content_with_weight`，供 embedding 和后续生成式问答使用。
    """

    rendered_parts: list[str] = []
    last_section = ""
    for unit in units:
        section_prefix = ""
        # 只有小节发生变化时才重复打 section 前缀，避免正文里到处重复。
        if unit["section_path"] and unit["section_path"] != last_section:
            section_prefix = f"小节：{unit['section_path']}\n"
            last_section = unit["section_path"]
        rendered_parts.append(section_prefix + unit["text"])

    merged_text = "\n\n".join(rendered_parts).strip()
    section_path = units[-1]["section_path"] if units else chapter_root
    content_with_weight = "\n".join(
        [
            f"来源文档：{source.title}",
            f"专题：{source.specialty}",
            f"章节：{section_path}",
            f"内容：{merged_text}",
        ]
    ).strip()
    # block_types 是一个很有用的调试字段，
    # 后续如果想分析“表格 chunk 的召回效果”和“段落 chunk 的召回效果”会很方便。
    block_types = sorted({unit["block_type"] for unit in units})
    row = {
        "pdf_title": source.title,
        "source_file": source.source_name,
        "source_md5": source.source_md5,
        "specialty": source.specialty,
        "chapter_root": chapter_root,
        "section_path": section_path,
        "block_types": ",".join(block_types),
        "table_count": sum(unit["table_count"] for unit in units),
        "search_text": f"{source.title} {source.specialty} {section_path} {merged_text}".strip(),
        "content_text": merged_text,
        "content_with_weight": content_with_weight,
        "chunk_token_estimate": num_tokens_from_string(content_with_weight),
    }
    return row


def merge_units_to_chunk_rows(
    source: PdfSourceInfo,
    chapter_root: str,
    units: list[dict],
    chunk_token_num: int,
    overlap_tokens: int,
) -> list[dict]:
    """把单元合并成最终 chunk 行。

    这是“细粒度内容单元 -> 检索 chunk”的关键步骤。
    规则是：
    - 尽量不超过 `chunk_token_num`
    - 如有需要，可从上一 chunk 尾部带少量 overlap
    """

    rows: list[dict] = []
    current_units: list[dict] = []
    current_tokens = 0

    def flush_current() -> list[dict]:
        nonlocal current_units, current_tokens
        if not current_units:
            return []
        row = finalize_chunk_row(source, chapter_root, current_units)
        rows.append(row)
        flushed = current_units
        current_units = []
        current_tokens = 0
        return flushed

    for unit in units:
        unit_tokens = unit["token_estimate"]
        # 新单元再放进去就超预算时，先把当前 chunk 落盘，再开启下一个 chunk。
        if current_units and current_tokens + unit_tokens > chunk_token_num:
            flushed_units = flush_current()
            current_units = tail_units_for_overlap(flushed_units, overlap_tokens)
            current_tokens = sum(item["token_estimate"] for item in current_units)

        current_units.append(unit)
        current_tokens += unit_tokens

    flush_current()
    return rows


def write_chunk_rows_as_shards(
    source: PdfSourceInfo,
    chapter_root: str,
    chunk_rows: list[dict],
    workdir: Path,
    logical_doc_token_num: int,
    logical_doc_max_chunks: int,
    part_index_start: int,
) -> tuple[list[ShardPlan], int]:
    """把同一章下的 chunk 进一步切成 logical document 分片。

    注意这里的目标不是再次切“检索 chunk”，而是切“document 管理边界”：
    - chunk 是检索单位
    - logical document 是入库/续跑/管理单位
    """

    shard_plans: list[ShardPlan] = []
    current_rows: list[dict] = []
    current_tokens = 0
    part_index = part_index_start

    def flush_rows() -> None:
        nonlocal current_rows, current_tokens, part_index
        if not current_rows:
            return
        shard_name = make_shard_name(source, chapter_root, part_index)
        shard_path = workdir / "normalized_shards" / f"{shard_name}.jsonl"
        rows_to_write: list[dict] = []
        # 每条 JSONL 记录都补上“自己在 shard 内的顺序”，
        # import 阶段会继续把它映射成 ES chunk 的顺序字段。
        for row_index, row in enumerate(current_rows, start=1):
            row_copy = dict(row)
            row_copy["chunk_index_in_shard"] = row_index
            rows_to_write.append(row_copy)
        write_jsonl(shard_path, rows_to_write)
        # 这里立刻计算 JSONL 文件自身的 md5/size，
        # 后面 import 阶段就可以拿它做幂等校验。
        raw_bytes = shard_path.read_bytes()
        shard_plans.append(
            ShardPlan(
                source_name=source.source_name,
                source_md5=source.source_md5,
                page_count=source.page_count,
                title=source.title,
                specialty=source.specialty,
                chapter_root=chapter_root,
                section_path=current_rows[-1]["section_path"],
                part_index=part_index,
                shard_name=shard_name,
                shard_file=str(shard_path.relative_to(workdir)).replace("\\", "/"),
                planned_count=len(rows_to_write),
                token_estimate=sum(row["chunk_token_estimate"] for row in rows_to_write),
                file_md5=hashlib.md5(raw_bytes).hexdigest(),
                file_size=len(raw_bytes),
            )
        )
        part_index += 1
        current_rows = []
        current_tokens = 0

    for row in chunk_rows:
        row_tokens = row["chunk_token_estimate"]
        # logical document 的切分依据是“双阈值”：
        # 1. chunk 数量不要太多
        # 2. token 总量不要太大
        if current_rows and (
            len(current_rows) >= logical_doc_max_chunks
            or current_tokens + row_tokens > logical_doc_token_num
        ):
            flush_rows()
        current_rows.append(row)
        current_tokens += row_tokens

    flush_rows()
    return shard_plans, part_index


def materialize_source(
    source: PdfSourceInfo,
    blocks: list[dict],
    workdir: Path,
    chunk_token_num: int,
    chunk_overlap_tokens: int,
    logical_doc_token_num: int,
    logical_doc_max_chunks: int,
    heading_split_depth: int,
) -> tuple[list[ShardPlan], dict]:
    """对单个 PDF 生成 logical documents。

    这一步已经不再关心 MinerU 如何工作，
    它只关心“给定一组 blocks，怎么切成适合入库的 shards”。
    """

    units = blocks_to_units(source, blocks, heading_split_depth)
    # 没有有效单元就没有可导入内容。
    if not units:
        return [], {
            "block_count": len(blocks),
            "table_count": sum(1 for block in blocks if block["type"] == "table"),
            "chunk_count": 0,
            "shard_count": 0,
        }

    chapter_groups: list[tuple[str, list[dict]]] = []
    current_chapter = ""
    current_units: list[dict] = []
    # 先按“章节根”做一层分组。
    # 这样不同大章天然不会被合并进同一个 logical document。
    for unit in units:
        chapter_root = unit["chapter_root"] or source.specialty or source.title
        if current_units and chapter_root != current_chapter:
            chapter_groups.append((current_chapter, current_units))
            current_units = []
        current_chapter = chapter_root
        current_units.append(unit)
    if current_units:
        chapter_groups.append((current_chapter, current_units))

    shard_plans: list[ShardPlan] = []
    chunk_count = 0
    chapter_part_counter: Counter[str] = Counter()
    # 每个章节组先切成 chunk，再组装成 logical document shards。
    for chapter_root, chapter_units in chapter_groups:
        chunk_rows = merge_units_to_chunk_rows(
            source,
            chapter_root,
            chapter_units,
            chunk_token_num,
            chunk_overlap_tokens,
        )
        chunk_count += len(chunk_rows)
        shard_rows, next_part_index = write_chunk_rows_as_shards(
            source,
            chapter_root,
            chunk_rows,
            workdir,
            logical_doc_token_num,
            logical_doc_max_chunks,
            chapter_part_counter[chapter_root] + 1,
        )
        chapter_part_counter[chapter_root] = next_part_index - 1
        shard_plans.extend(shard_rows)

    stats = {
        "block_count": len(blocks),
        "table_count": sum(1 for block in blocks if block["type"] == "table"),
        "chunk_count": chunk_count,
        "shard_count": len(shard_plans),
    }
    return shard_plans, stats


def materialize_shards(
    workdir: Path,
    pdf_sources: list[PdfSourceInfo],
    runner: MinerURunner,
    parse_method: str,
    chunk_token_num: int,
    chunk_overlap_tokens: int,
    logical_doc_token_num: int = DEFAULT_LOGICAL_DOC_TOKEN_NUM,
    logical_doc_max_chunks: int = DEFAULT_LOGICAL_DOC_MAX_CHUNKS,
    heading_split_depth: int = DEFAULT_HEADING_SPLIT_DEPTH,
    force_reparse: bool = False,
) -> tuple[dict, list[ShardPlan]]:
    """执行 MinerU 预处理、chunk 生成和 logical document 物化。

    它是整个 materialize 阶段的总调度器：
    - 负责目录初始化
    - 负责缓存加载与更新
    - 负责逐个 PDF 调用 materialize_source
    - 负责汇总统计信息
    """

    normalized_dir = workdir / "normalized_shards"
    if normalized_dir.exists():
        # 和 QA 版不同，MinerU 缓存不会删；
        # 这里只清理最终 JSONL 产物，保证 shard 和本轮计划一致。
        shutil.rmtree(normalized_dir)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    (workdir / "mineru_markdown").mkdir(parents=True, exist_ok=True)
    (workdir / "parsed_blocks").mkdir(parents=True, exist_ok=True)

    cache_path = workdir / "materialize_cache.json"
    cache_index = load_mineru_cache(cache_path)
    updated_cache: dict[str, MinerUCacheInfo] = {}

    reused_parse_count = 0
    fresh_parse_count = 0
    total_blocks = 0
    total_tables = 0
    total_chunks = 0
    all_plans: list[ShardPlan] = []
    source_summaries: list[dict] = []

    for source in pdf_sources:
        logging.info("Materializing PDF source: %s", source.source_name)
        blocks, reused, cache_info = load_or_parse_blocks(
            source=source,
            runner=runner,
            workdir=workdir,
            parse_method=parse_method,
            cache_index=cache_index,
            force_reparse=force_reparse,
        )
        # 无论本轮是复用还是重解析，都把最终 cache_info 回写，保证索引最新。
        updated_cache[source.source_name] = cache_info
        source_plans, stats = materialize_source(
            source=source,
            blocks=blocks,
            workdir=workdir,
            chunk_token_num=chunk_token_num,
            chunk_overlap_tokens=chunk_overlap_tokens,
            logical_doc_token_num=logical_doc_token_num,
            logical_doc_max_chunks=logical_doc_max_chunks,
            heading_split_depth=heading_split_depth,
        )
        all_plans.extend(source_plans)
        reused_parse_count += int(reused)
        fresh_parse_count += int(not reused)
        total_blocks += stats["block_count"]
        total_tables += stats["table_count"]
        total_chunks += stats["chunk_count"]
        source_summaries.append(
            {
                "source_name": source.source_name,
                "title": source.title,
                "specialty": source.specialty,
                "page_count": source.page_count,
                "file_size": source.file_size,
                "reused_parse": reused,
                **stats,
            }
        )

    # materialize 结束时统一刷新缓存索引。
    write_json(cache_path, {name: asdict(item) for name, item in updated_cache.items()})

    analysis = {
        "source_count": len(pdf_sources),
        "total_pages": sum(source.page_count for source in pdf_sources),
        "total_size": sum(source.file_size for source in pdf_sources),
        "reused_parse_count": reused_parse_count,
        "fresh_parse_count": fresh_parse_count,
        "total_blocks": total_blocks,
        "total_tables": total_tables,
        "total_chunks": total_chunks,
        "logical_doc_count": len(all_plans),
        "sources": source_summaries,
    }
    return analysis, all_plans
