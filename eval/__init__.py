"""Evaluation harness for the tutor's natural-language layer.

Importing this package configures the environment for a **deterministic** run:
no API key (so the keyword router and template answers are exercised), an
in-memory cache, and no scheduler. This happens at package import — before any
``app`` module is loaded — because ``get_settings()`` is an ``lru_cache``
singleton and patching it after the fact would leave the cached instance
pointing at the developer's real ``.env``.

Run it with ``python -m eval.run_eval``.
"""

from __future__ import annotations

import os

os.environ.setdefault("CODEFORCES_HANDLE", "EvalUser")
os.environ.setdefault("LEETCODE_USERNAME", "evaluser")
os.environ.update(
    {
        "ANTHROPIC_API_KEY": "",
        "LEETCODE_SESSION": "",
        "CACHE_BACKEND": "memory",
        "SYNC_ENABLED": "false",
        "SYNC_ON_STARTUP": "false",
        "LOG_LEVEL": "ERROR",
    }
)
