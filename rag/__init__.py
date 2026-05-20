"""RAG 子包入口。

这个目录承载了项目里的检索增强生成核心能力，包括：
- `nlp/`：分词、切块、查询构造、检索排序
- `llm/`：聊天、向量、重排等模型适配
- `prompts/`：Prompt 模板与知识拼接
- `utils/`：ES / Redis / MinIO / 图像等基础设施适配
- `app/`：不同类型知识库的切块实现
"""

#

# 如果后续需要对整个包启用运行时类型检查，可恢复下面两行：
# from beartype.claw import beartype_this_package
# beartype_this_package()
