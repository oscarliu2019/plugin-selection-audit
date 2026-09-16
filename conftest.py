"""让 `pytest` 从仓库根目录导入 `src` 包（prepend import 模式下会插入本文件所在目录）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
