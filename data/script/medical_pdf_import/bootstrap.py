"""运行时引导。

这个模块只做一件事：把项目根目录放进 `sys.path`。
这样 `data/script` 下的新脚本包可以直接复用项目里的公共基础设施，
例如 `api/`、`common/`、`rag/` 等模块。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
# 这里使用“只在不存在时插入”的方式，避免重复污染 sys.path，
# 同时保证脚本既可以从仓库根目录运行，也可以直接点某个文件运行。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
