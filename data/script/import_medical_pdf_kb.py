#!/usr/bin/env python3
"""
医疗 PDF 知识库导入脚本入口。

这个入口文件刻意保持很薄，只负责两件事：

1. 保留稳定的命令行调用方式
   `python data\\script\\import_medical_pdf_kb.py --phase materialize`
   `python data\\script\\import_medical_pdf_kb.py `
    --phase import `
    --kb-name medical_pdf_kb_v1 `
    --mineru-mode api `
    --mineru-api-url http://127.0.0.1:8003 `
    --mineru-parse-method auto`
2. 把真正实现委托给 `medical_pdf_import` 包
   这样 MinerU 预处理、分片物化、正式入库、报告生成可以分模块维护。
"""

from medical_pdf_import import main


if __name__ == "__main__":
    main()
