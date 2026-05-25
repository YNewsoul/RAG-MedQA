"""运行时引导。

这个模块只做一件事：确保项目根目录进入 `sys.path`。
这样拆分后的子模块依旧可以直接复用项目内部的 `api/`、`common/`、`rag/`
等模块，而不需要在每个文件里重复写一遍路径注入逻辑。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
