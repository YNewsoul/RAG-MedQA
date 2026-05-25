"""正式入库逻辑。

这个模块负责把已经物化好的 JSONL 分片写入：
- MySQL 中的 knowledgebase / document / file / file2document
- ES 主索引 `ragmedqa`
- 文档 metadata 索引 `ragmedqa_doc_meta`
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .bootstrap import PROJECT_ROOT
from .io_utils import load_jsonl
from .schema import (
    deterministic_chunk_id,
    deterministic_doc_id,
    MAJOR_ORDER,
    ShardPlan,
)

from api.db import FileType  # noqa: E402
from api.db.db_models import Knowledgebase, init_database_tables  # noqa: E402
from api.db.joint_services.tenant_model_service import (  # noqa: E402
    get_tenant_default_model_by_type,
)
from api.db.services.doc_metadata_service import DocMetadataService  # noqa: E402
from api.db.services.document_service import DocumentService  # noqa: E402
from api.db.services.file2document_service import File2DocumentService  # noqa: E402
from api.db.services.file_service import FileService  # noqa: E402
from api.db.services.knowledgebase_service import KnowledgebaseService  # noqa: E402
from api.db.services.llm_service import LLMBundle  # noqa: E402
from api.utils.api_utils import get_parser_config  # noqa: E402
from common import settings  # noqa: E402
from common.constants import (  # noqa: E402
    LLMType,
    ParserType,
    SYSTEM_TENANT_ID,
    StatusEnum,
    TaskStatus,
)
from common.misc_utils import get_uuid  # noqa: E402
from rag.nlp import rag_tokenizer, search  # noqa: E402


def ensure_runtime() -> None:
    """初始化运行时环境。

    import 阶段真正需要访问：
    - MySQL
    - ES
    - embedding 模型

    所以在这里先做项目级 settings 初始化和建表初始化。
    """

    settings.init_settings()
    init_database_tables()


def ensure_knowledge_base(kb_name: str) -> Knowledgebase:
    """创建或复用 QA 知识库。"""

    kb = Knowledgebase.get_or_none(
        (Knowledgebase.name == kb_name) & (Knowledgebase.status == StatusEnum.VALID.value)
    )
    if kb is not None:
        return kb

    # 这里预先把 metadata schema 写进 parser_config，方便后续从后台理解这些分片。
    parser_config = get_parser_config(
        ParserType.QA.value,
        {
            "enable_metadata": True,
            "metadata": [
                {"name": "data_type", "type": "string", "description": "固定为 qa"},
                {"name": "major_category", "type": "string", "description": "一级医学大类"},
                {"name": "department", "type": "string", "description": "二级分片科室或 misc"},
                {"name": "bucket_type", "type": "string", "description": "clean_department 或 long_tail"},
                {"name": "source_file", "type": "string", "description": "来源原始 JSON 文件"},
                {"name": "shard_name", "type": "string", "description": "逻辑文档分片名"},
            ],
        },
    )
    parser_config["llm_id"] = settings.CHAT_MDL or ""

    payload = {
        "id": get_uuid(),
        "name": kb_name,
        "language": "Chinese",
        "description": "医疗 QA 知识库（由 data/script/import_medical_qa_kb.py 导入）",
        "embd_id": settings.EMBEDDING_MDL,
        "permission": "me",
        "created_by": SYSTEM_TENANT_ID,
        "parser_id": ParserType.QA.value,
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
    """确保 file/file2document 层存在该知识库对应的目录。"""

    root_folder = FileService.get_root_folder(SYSTEM_TENANT_ID)
    kb_root = FileService.get_kb_folder(SYSTEM_TENANT_ID)
    if not kb_root:
        FileService.init_knowledgebase_docs(root_folder["id"], SYSTEM_TENANT_ID)
        kb_root = FileService.get_kb_folder(SYSTEM_TENANT_ID)
    return FileService.new_a_file_from_kb(SYSTEM_TENANT_ID, kb_name, kb_root["id"])


def existing_document_is_usable(doc, expected_hash: str) -> bool:
    """判断已有 document 是否可以直接复用。

    这里体现的是脚本的“分片级幂等”设计：
    - 只要 document 的内容哈希没变
    - chunk 数、状态、进度都完整
    - file/file2document 映射也还在

    那么这一个分片就视为“已经完整导入过”，后续 rerun 时会直接跳过。
    """

    return (
        doc.content_hash == expected_hash
        and doc.chunk_num > 0
        and str(doc.run) == TaskStatus.DONE.value
        and float(doc.progress or 0) == 1.0
        and bool(File2DocumentService.get_by_document_id(doc.id))
    )


def cleanup_document(doc_id: str) -> None:
    """清理一个已存在但不可信的 document。

    典型触发场景是：
    - 上一次导入中途断网
    - embedding 或 ES 写入时抛错
    - document 主记录写进去了，但 chunk 没写完整

    这时不能简单复用旧记录，否则会留下“表面成功、实际缺块”的脏状态。
    所以这里会把 document、本地映射关系，以及关联 file 记录一起清掉，
    让这个 shard 可以按一个全新的分片重新导入。
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

    注意这里的 document 不是原始大 JSON 文件，而是一个 logical document，
    即我们 materialize 阶段生成的某个 `qa_xxx_pNNN.jsonl` 分片。
    """

    doc_id = deterministic_doc_id(kb.id, plan.shard_name)
    doc_name = f"{plan.shard_name}.jsonl"
    shard_path = workdir / plan.shard_file
    try:
        rel_location = str(shard_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        rel_location = str(shard_path).replace("\\", "/")

    return {
        "id": doc_id,
        "kb_id": kb.id,
        "parser_id": ParserType.QA.value,
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
        "progress_msg": "Imported by data/script/import_medical_qa_kb.py",
        "process_duration": 0,
        "suffix": "jsonl",
        "run": TaskStatus.DONE.value,
        "status": StatusEnum.VALID.value,
        "content_hash": plan.file_md5,
    }


def shard_metadata(plan: ShardPlan) -> dict:
    """构造 logical document 的 metadata。

    这部分元数据会写进文档 metadata 索引，用来帮助后台理解：
    - 这个分片属于哪个一级大类
    - 是哪个 bucket
    - 来自哪个源文件
    - 在整个分片体系里叫什么名字
    """

    return {
        "data_type": "qa",
        "major_category": plan.major_category,
        "department": plan.bucket_department,
        "bucket_type": plan.bucket_type,
        "source_file": plan.source_file,
        "shard_name": plan.shard_name,
        "planned_count": plan.planned_count,
    }


def chunk_from_record(doc_id: str, doc_name: str, kb_id: str, row: dict, row_index: int) -> dict:
    """把单条 QA 记录转换成 ES chunk。

    两类字段要特别留意：

    - `question_tks` / `content_ltks`
      偏向全文检索的“问题面”。
    - `content_with_weight`
      偏向向量召回和最终引用展示的“完整内容面”。
    """

    question_text = row["question_text"]
    title_text = row["title"] or doc_name
    content_ltks = rag_tokenizer.tokenize(question_text)

    return {
        "id": deterministic_chunk_id(doc_id, row["ask"], row["answer"]), # 为单条 QA 生成稳定的 chunk id
        "doc_id": doc_id,
        "kb_id": kb_id,
        "docnm_kwd": doc_name, # 文档名称，用于检索
        "title_tks": rag_tokenizer.tokenize(title_text), # 标题，用于检索
        "question_kwd": question_text,
        "question_tks": rag_tokenizer.tokenize(question_text), # 问题，用于检索
        "content_ltks": content_ltks, # 内容，用于检索
        "content_sm_ltks": rag_tokenizer.fine_grained_tokenize(content_ltks), # 内容，用于检索
        "content_with_weight": row["content_with_weight"],
        "important_kwd": [],
        "doc_type_kwd": "qa",
        "available_int": 1,
        "chunk_order_int": row_index,
        "top_int": [row_index],
        "create_time": str(datetime.now()).replace("T", " ")[:19],
        "create_timestamp_flt": datetime.now().timestamp(), # 创建时间，用于排序
    }


def import_shards(
    kb: Knowledgebase,
    kb_folder: dict,
    workdir: Path,
    shard_plans: list[ShardPlan],
    embed_batch_size: int,
    insert_batch_size: int,
) -> list[dict]:
    """把 JSONL 分片真正写入知识库、MySQL 和 ES。

    这是脚本真正“落地数据”的阶段。每个 shard 的处理顺序固定为：

    1. 检查这个分片是否已经成功导入过
    2. 如有半成品旧数据，则先清理
    3. 写 document 主记录
    4. 写 file / file2document 映射
    5. 写文档级 metadata
    6. 读取 JSONL，逐条转成 chunk
    7. 批量请求 embedding
    8. 批量写入 ES
    9. 回写 chunk_num / token_num

    其中任何一步失败，当前 shard 都会回滚到“未导入”状态，
    这样 rerun 时才能稳定续跑。
    """

    embed_cfg = get_tenant_default_model_by_type(LLMType.EMBEDDING)
    emb_model = LLMBundle(embed_cfg)
    idx_name = search.index_name()  # 索引仓库名称，意思是构建一个叫ragmedqa的索引仓库
    import_results: list[dict] = []  # 导入结果
    index_ready = settings.docStoreConn.index_exist(idx_name, kb.id)  # 索引是否存在

    for plan in shard_plans:
        shard_path = workdir / plan.shard_file
        doc_payload = build_document_payload(kb, plan, workdir)
        doc_id = doc_payload["id"]
        plan.doc_id = doc_id

        # 如果同内容 shard 之前已完整导入，则直接跳过。
        ok, existing_doc = DocumentService.get_by_id(doc_id)
        if ok and existing_document_is_usable(existing_doc, plan.file_md5):
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

        # 如果旧 document 存在但状态不可信，先整片清掉后重建。
        if ok:
            cleanup_document(doc_id)

        try:
            # 先写 document 与文件映射，再写 metadata，再写 ES chunk。
            # 这样做的目的是让平台侧“先看见这个 logical document 的存在”，
            # 再逐步补齐它的检索内容。
            DocumentService.insert(doc_payload)
            # 构建文件目录结构
            FileService.add_file_from_kb(doc_payload, kb_folder["id"], SYSTEM_TENANT_ID)
            if not DocMetadataService.insert_document_metadata(doc_id, shard_metadata(plan)):
                raise RuntimeError(f"Failed to insert metadata for shard {plan.shard_name}")

            # 读取 JSONL 文件，逐条转换成 chunk。
            rows = load_jsonl(shard_path)
            chunks = [
                chunk_from_record(doc_id, doc_payload["name"], kb.id, row, row_index)
                for row_index, row in enumerate(rows, start=1)
            ]

            token_total = 0
            chunk_total = 0

            # embedding 分批做，避免一次性提交太多文本给向量模型。
            for start in range(0, len(chunks), embed_batch_size):
                batch = chunks[start : start + embed_batch_size]
                texts = [item["content_with_weight"] for item in batch]
                vectors, used_tokens = emb_model.encode(texts)  # 拿到向量表示
                token_total += used_tokens
                chunk_total += len(batch)

                normalized_vectors = []
                for vector in vectors:
                    # 向量模型返回的 vector 可能是 numpy 数组，也可能是 list。
                    if hasattr(vector, "tolist"):
                        normalized_vectors.append(vector.tolist())
                    else:
                        normalized_vectors.append(list(vector))

                # 第一次真正写入 chunk 时，如果 ES 索引还不存在，就按向量维度先建索引。
                if normalized_vectors and not index_ready:
                    vector_dimension = len(normalized_vectors[0])
                    # docStoreConn：rag.utils.es_conn.ESConnection()，即ES连接对象
                    settings.docStoreConn.create_idx(idx_name, kb.id, vector_dimension, kb.parser_id)
                    index_ready = True

                ready_docs = []
                for chunk, vector in zip(batch, normalized_vectors):
                    chunk[f"q_{len(vector)}_vec"] = vector
                    ready_docs.append(chunk)

                # bulk 写 ES 继续分批，降低单次请求体积。
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

            # 全部 chunk 成功后，再回写 token_num 与 chunk_num。
            # 这一步必须放在最后，避免中途失败时 document 被误判为完整可复用。
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
            # 单 shard 失败时，回滚 document 侧状态，避免留下半成功脏数据。
            cleanup_document(doc_id)
            raise

    return import_results
