"""医疗 PDF 知识库导入脚本包。

这里只暴露两个最稳定的对外符号：

- `PROJECT_ROOT`
  供外部定位项目根目录时复用
- `main`
  供入口脚本直接调用

这样后续包内部即便继续拆文件，入口层也不需要感知细节变化。
"""

from .bootstrap import PROJECT_ROOT  # noqa: F401
from .cli import main

__all__ = ["PROJECT_ROOT", "main"]
