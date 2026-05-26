"""PDF logical documents 的正式入库逻辑。

这个模块只处理“标准化 JSONL -> MySQL/ES”的部分，
不再触碰原始 PDF，也不关心 MinerU。

这样做的好处是：
1. import-only 可以稳定续跑
2. 入库逻辑和 PDF 解析逻辑彻底解耦
3. materialize 与 import 可以独立调试
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .bootstrap import PROJECT_ROOT
from .io_utils import load_jsonl
from .schema import ShardPlan, deterministic_chunk_id, deterministic_doc_id

from api.db import FileType  # noqa: E402
from api.db.db_models import Knowledgebase, init_database_tables  # noqa: E402
from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type  # noqa: E402
from api.db.services.doc_metadata_service import DocMetadataService  # noqa: E402
from api.db.services.document_service import DocumentService  # noqa: E402
from api.db.services.file2document_service import File2DocumentService  # noqa: E402
from api.db.services.file_service import FileService  # noqa: E402
from api.db.services.knowledgebase_service import KnowledgebaseService  # noqa: E402
from api.db.services.llm_service import LLMBundle  # noqa: E402
from api.utils.api_utils import get_parser_config  # noqa: E402
from common import settings  # noqa: E402
from common.constants import LLMType, ParserType, SYSTEM_TENANT_ID, StatusEnum, TaskStatus  # noqa: E402
from common.misc_utils import get_uuid  # noqa: E402
from rag.nlp import rag_tokenizer, search  # noqa: E402


def ensure_runtime() -> None:
    """初始化运行时环境。

    import 阶段会真正访问：
    - MySQL
    - Elasticsearch
    - embedding 服务

    所以这里统一做项目级 settings 初始化和建表初始化。
    """

    settings.init_settings()
    init_database_tables()


def ensure_knowledge_base(kb_name: str) -> Knowledgebase:
    """创建或复用医疗 PDF 知识库。

    这里定义的是“这套 PDF KB 的默认元数据 schema”和 parser 身份。
    即便当前脚本没有直接走项目原有 PDF chunker，
    仍然要让数据库里的 KB 在语义上是一个“book 型 PDF 知识库”。
    """

    kb = Knowledgebase.get_or_none(
        (Knowledgebase.name == kb_name) & (Knowledgebase.status == StatusEnum.VALID.value)
    )
    if kb is not None:
        return kb

    parser_config = get_parser_config(
        ParserType.BOOK.value,
        {
            "layout_recognize": "MinerU",
            "enable_metadata": True,
            "metadata": [
                {"name": "data_type", "type": "string", "description": "固定为 pdf"},
                {"name": "source_file", "type": "string", "description": "来源 PDF 文件名"},
                {"name": "pdf_title", "type": "string", "description": "PDF 标题"},
                {"name": "specialty", "type": "string", "description": "医学专题"},
                {"name": "chapter_root", "type": "string", "description": "逻辑文档所属章节根"},
                {"name": "section_path", "type": "string", "description": "当前分片主要章节路径"},
                {"name": "source_md5", "type": "string", "description": "来源 PDF 的 MD5"},
            ],
        },
    )
    # 给 parser_config 填一个默认聊天模型，便于后台侧后续读取这份配置时更完整。
    parser_config["llm_id"] = settings.CHAT_MDL or ""

    payload = {
        "id": get_uuid(),
        "name": kb_name,
        "language": "Chinese",
        "description": "医疗 PDF 知识库（由 data/script/import_medical_pdf_kb.py 导入）",
        "embd_id": settings.EMBEDDING_MDL,
        "permission": "me",
        "created_by": SYSTEM_TENANT_ID,
        "parser_id": ParserType.BOOK.value,
        "parser_config": parser_config,
        "status": StatusEnum.VALID.value,
    }
    if not KnowledgebaseService.save(**payload):
        raise RuntimeError(f"Failed to create knowledge base: {kb_name}")

    kb = Knowledgebase.get_or_none(Knowledgebase.id == payload["id"])
    if kb is None:
        raise RuntimeError(f"Failed to fetch knowledge base after creation: {kb_name}")
    return kb


def ensure_kb_folder(kb_name: str) -> dict:
    """确保 file / file2document 层存在当前知识库目录。

    这不是物理文件夹，而是数据库里那棵虚拟文件树。
    它的存在可以让导入的 logical documents 在平台文件视图里也可见。
    """

    root_folder = FileService.get_root_folder(SYSTEM_TENANT_ID)
    kb_root = FileService.get_kb_folder(SYSTEM_TENANT_ID)
    if not kb_root:
        FileService.init_knowledgebase_docs(root_folder["id"], SYSTEM_TENANT_ID)
        kb_root = FileService.get_kb_folder(SYSTEM_TENANT_ID)
    return FileService.new_a_file_from_kb(SYSTEM_TENANT_ID, kb_name, kb_root["id"])


def existing_document_is_usable(doc, expected_hash: str) -> bool:
    """判断旧 document 是否可直接复用。

    这是 import 续跑的关键判断。
    只有当：
    - 内容哈希没变
    - chunk 数正常
    - 状态是 DONE
    - 进度是 1.0
    - file 映射还在

    才认为这片已经完整导入过。
    """

    return (
        doc.content_hash == expected_hash
        and doc.chunk_num > 0
        and str(doc.run) == TaskStatus.DONE.value
        and float(doc.progress or 0) == 1.0
        and bool(File2DocumentService.get_by_document_id(doc.id))
    )


def cleanup_document(doc_id: str) -> None:
    """删除一个不完整的 logical document。

    典型触发场景：
    - embedding 中途失败
    - ES bulk 插入失败
    - 只写进了 document 记录，还没把 chunk 全写完

    这时不能简单覆盖，而要先清理干净再重导，避免留下半成品。
    """

    ok, doc = DocumentService.get_by_id(doc_id)
    if not ok:
        return

    relations = File2DocumentService.get_by_document_id(doc_id)
    file_ids = [relation.file_id for relation in relations if relation.file_id]

    DocumentService.remove_document(doc, SYSTEM_TENANT_ID)
    File2DocumentService.delete_by_document_id(doc_id)

    for file_id in file_ids:
        FileService.delete_by_id(file_id)


def build_document_payload(kb: Knowledgebase, plan: ShardPlan, workdir: Path) -> dict:
    """构造 document 表写入 payload。

    注意这里的 document 指的是：
    “一个 PDF logical document 分片”
    而不是原始整本 PDF。
    """

    doc_id = deterministic_doc_id(kb.id, plan.shard_name)
    doc_name = f"{plan.shard_name}.jsonl"
    shard_path = workdir / plan.shard_file
    # 优先写相对项目根目录的 location，
    # 这样换机器或复制工作区时可读性更好。
    try:
        rel_location = str(shard_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        rel_location = str(shard_path).replace("\\", "/")

    return {
        "id": doc_id,
        "kb_id": kb.id,
        "parser_id": ParserType.BOOK.value,
        "parser_config": kb.parser_config,
        "source_type": "local",
        "type": FileType.OTHER.value,
        "created_by": SYSTEM_TENANT_ID,
        "name": doc_name,
        "location": rel_location,
        "size": plan.file_size,
        "token_num": 0,
        "chunk_num": 0,
        "progress": 1.0,
        "progress_msg": "Imported by data/script/import_medical_pdf_kb.py",
        "process_duration": 0,
        "suffix": "jsonl",
        "run": TaskStatus.DONE.value,
        "status": StatusEnum.VALID.value,
        "content_hash": plan.file_md5,
    }


def shard_metadata(plan: ShardPlan) -> dict:
    """构造 logical document 元数据。

    这部分会进入 `ragmedqa_doc_meta` 元数据索引，
    便于后续按专题、章节、来源 PDF 做过滤和排查。
    """

    return {
        "data_type": "pdf",
        "source_file": plan.source_name,
        "pdf_title": plan.title,
        "specialty": plan.specialty,
        "chapter_root": plan.chapter_root,
        "section_path": plan.section_path,
        "source_md5": plan.source_md5,
    }


def chunk_from_record(doc_id: str, doc_name: str, kb_id: str, row: dict, row_index: int) -> dict:
    """把单条 PDF chunk 记录转成 ES 文档。

    这里构造的是“最终写进 ES 的一条记录”：
    - `_kwd` 字段用于过滤/精确匹配
    - `_tks` / `_ltks` 字段用于全文检索
    - `content_with_weight` 用于 embedding 和最终展示
    """

    # search_text 是偏检索视角的文本；
    # content_with_weight 则是偏语义表达和引用展示的文本。
    search_text = row["search_text"]
    content_ltks = rag_tokenizer.tokenize(search_text)

    return {
        "id": deterministic_chunk_id(doc_id, row_index, row["content_with_weight"]),
        "doc_id": doc_id,
        "kb_id": kb_id,
        "docnm_kwd": doc_name,
        "title_tks": rag_tokenizer.tokenize(row["pdf_title"]),
        "source_file_kwd": row["source_file"],
        "specialty_kwd": row["specialty"],
        "chapter_root_kwd": row["chapter_root"],
        "section_path_kwd": row["section_path"],
        "content_ltks": content_ltks,
        "content_sm_ltks": rag_tokenizer.fine_grained_tokenize(content_ltks),
        "content_with_weight": row["content_with_weight"],
        "important_kwd": [],
        "doc_type_kwd": "pdf",
        "table_count_int": int(row.get("table_count", 0) or 0),
        "available_int": 1,
        "chunk_order_int": row_index,
        "top_int": [row_index],
        "create_time": str(datetime.now()).replace("T", " ")[:19],
        "create_timestamp_flt": datetime.now().timestamp(),
    }


def import_shards(
    kb: Knowledgebase,
    kb_folder: dict,
    workdir: Path,
    shard_plans: list[ShardPlan],
    embed_batch_size: int,
    insert_batch_size: int,
) -> list[dict]:
    """把 materialize 产出的 logical documents 正式写入 KB / ES。

    处理顺序固定为：
    1. 检查 shard 是否已完整导入
    2. 如有旧半成品则清理
    3. 写 document / file / metadata
    4. 读 JSONL，逐条变成 ES chunk
    5. 做 embedding
    6. 写 ES
    7. 回写 token / chunk 统计
    """

    # 这套 PDF KB 和 QA KB 一样，统一复用全局默认 embedding 模型。
    embed_cfg = get_tenant_default_model_by_type(LLMType.EMBEDDING)
    emb_model = LLMBundle(embed_cfg)
    idx_name = search.index_name()  # 索引名称，用于写入 ES
    import_results: list[dict] = []
    # 如果主索引已经存在，后面就不再重复 create。
    index_ready = settings.docStoreConn.index_exist(idx_name, kb.id)

    for plan in shard_plans:
        shard_path = workdir / plan.shard_file
        doc_payload = build_document_payload(kb, plan, workdir)  # 构建 document payload
        doc_id = doc_payload["id"]  # 提取 document ID
        plan.doc_id = doc_id  # 记录 document ID

        # 先判断这个 shard 是否早就成功导入过。
        ok, existing_doc = DocumentService.get_by_id(doc_id)
        if ok and existing_document_is_usable(existing_doc, plan.file_md5):
            # 如果旧 document 在，且状态可信，就直接跳过。
            import_results.append(
                {
                    "shard_name": plan.shard_name,
                    "doc_id": doc_id,
                    "status": "skipped_existing",
                    "chunk_count": existing_doc.chunk_num,
                    "token_num": existing_doc.token_num,
                }
            )
            continue

        # 如果旧 document 在，但状态不可信，就整片删掉重来。
        if ok:
            cleanup_document(doc_id)

        try:
            # 先把“管理层可见”的对象写进去，再补检索层内容。
            DocumentService.insert(doc_payload)
            FileService.add_file_from_kb(doc_payload, kb_folder["id"], SYSTEM_TENANT_ID)
            if not DocMetadataService.insert_document_metadata(doc_id, shard_metadata(plan)):
                raise RuntimeError(f"Failed to insert metadata for shard {plan.shard_name}")

            # 读取标准化 JSONL，再逐条构造成 ES chunk。
            rows = load_jsonl(shard_path)
            chunks = [
                chunk_from_record(doc_id, doc_payload["name"], kb.id, row, row_index)
                for row_index, row in enumerate(rows, start=1)
            ]

            token_total = 0
            chunk_total = 0

            # embedding 分批做，避免一次性压爆远端 embedding 服务。
            for start in range(0, len(chunks), embed_batch_size):
                batch = chunks[start : start + embed_batch_size]
                texts = [item["content_with_weight"] for item in batch]
                vectors, used_tokens = emb_model.encode(texts)
                token_total += used_tokens
                chunk_total += len(batch)

                normalized_vectors = []
                # 某些 embedding 客户端返回 numpy，某些返回 list；
                # 这里统一归一化成普通 Python list。
                for vector in vectors:
                    if hasattr(vector, "tolist"):
                        normalized_vectors.append(vector.tolist())
                    else:
                        normalized_vectors.append(list(vector))

                # 第一次真正写 chunk 前，如 ES 主索引还不存在，就先按向量维度建索引。
                if normalized_vectors and not index_ready:
                    vector_dimension = len(normalized_vectors[0])
                    settings.docStoreConn.create_idx(idx_name, kb.id, vector_dimension, kb.parser_id)
                    index_ready = True

                ready_docs = []
                # 把向量挂回 chunk，形成最终可写入 ES 的完整文档。
                for chunk, vector in zip(batch, normalized_vectors):
                    chunk[f"q_{len(vector)}_vec"] = vector
                    ready_docs.append(chunk)

                # ES bulk 插入也继续分批，避免单次请求体过大。
                for insert_start in range(0, len(ready_docs), insert_batch_size):
                    errors = settings.docStoreConn.insert(
                        ready_docs[insert_start : insert_start + insert_batch_size],
                        idx_name,
                        kb.id,
                    )
                    if errors:
                        raise RuntimeError(
                            f"ES insert failed for shard {plan.shard_name}: {errors[:3]}"
                        )

            # 只有全部 chunk 成功写入后，才回写统计值。
            # 这一步必须放最后，否则半成品 document 会被误判为“已完成”。
            DocumentService.increment_chunk_num(doc_id, kb.id, token_total, chunk_total, 0)
            import_results.append(
                {
                    "shard_name": plan.shard_name,
                    "doc_id": doc_id,
                    "status": "imported",
                    "chunk_count": chunk_total,
                    "token_num": token_total,
                }
            )
        except Exception:
            # 当前 shard 失败时，立刻回滚 document 侧状态，保证 rerun 时能干净重来。
            cleanup_document(doc_id)
            raise

    return import_results
