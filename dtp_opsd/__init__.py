"""DTP-OPSD: Distractor-aware Temporal-Privilege On-Policy Self-Distillation."""

import os
import sys

# The GRPO base trainer comes from TimeLens. A minimal copy is vendored under
# third_party/TimeLens; set TIMELENS_ROOT to use an external checkout instead.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMELENS_ROOT = os.environ.get("TIMELENS_ROOT") or os.path.join(_REPO_ROOT, "third_party", "TimeLens")
if os.path.isdir(_TIMELENS_ROOT) and _TIMELENS_ROOT not in sys.path:
    sys.path.insert(0, _TIMELENS_ROOT)

__version__ = "0.1.0"
