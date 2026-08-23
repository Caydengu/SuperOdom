from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "g1_root_state_bridge"
sys.path.insert(0, str(PACKAGE_ROOT))
