#!/usr/bin/env python3
"""
医疗 QA 知识库导入脚本入口。

这个文件现在故意保持很薄，只承担两件事：

1. 作为稳定的命令行入口保留下来
   `python data\script\import_medical_qa_kb.py ...`
   `python data\script\import_medical_qa_kb.py --phase materialize`
   `python data\script\import_medical_qa_kb.py --phase import --kb-name medical_qa_kb_v1 --workdir data\script\output\medical_qa_import`
2. 把真正实现委托给 `medical_qa_import` 包
   这样分片规划、JSONL 物化、正式入库、报告生成这些职责
   就可以分别放到多个模块里维护，不再挤在一个超长脚本中。

内部模块划分如下：

- `medical_qa_import.schema`
  常量、dataclass、纯函数工具
- `medical_qa_import.io_utils`
  本地文件读写
- `medical_qa_import.planning`
  扫描原始 QA、生成分片计划、物化 JSONL
- `medical_qa_import.importing`
  写 MySQL / ES / metadata 的正式导入逻辑
- `medical_qa_import.reporting`
  Markdown 报告生成
- `medical_qa_import.cli`
  参数解析与主流程调度
"""

from medical_qa_import import main


if __name__ == "__main__":
    main()
