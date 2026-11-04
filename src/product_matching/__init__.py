"""Product matching, attribute extraction and search for marketplace listings."""

import sys

__version__ = "2.0.0"

if sys.platform == "darwin":  # pragma: no cover - platform-specific
    # On macOS, torch and xgboost each bundle their own OpenMP runtime. If torch is
    # loaded first, fitting an XGBoost model segfaults. Loading xgboost first avoids it.
    try:
        import xgboost  # noqa: F401
    except ImportError:
        pass
