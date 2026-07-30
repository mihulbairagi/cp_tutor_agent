"""Structured logging setup.

One `setup_logging()` call at process start configures the whole tree; every
module then does `logger = logging.getLogger(__name__)` and nothing else. No
`print()` anywhere in the codebase.

Two formatters are provided:

* **plain** — readable during local development.
* **json** — one JSON object per line, which is what log shippers (CloudWatch,
  Loki, Datadog) expect. Flipped on with `LOG_JSON=true`, so the same image
  behaves correctly in a container without a code change.
"""

from __future__ import annotations

import json
import logging
import logging.config
from typing import Any

# Attributes present on every LogRecord. Anything *not* in this set was added by
# the caller via `extra={...}`, so the JSON formatter promotes it to a top-level
# field. That is what makes logs queryable ("show me every sync where
# source=codeforces and fetched > 100").
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure the root logger and quiet down noisy third-party libraries."""
    formatter = "json" if json_output else "plain"

    logging.config.dictConfig(
        {
            "version": 1,
            # Leave module-level loggers created at import time attached.
            "disable_existing_loggers": False,
            "formatters": {
                "plain": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%H:%M:%S",
                },
                "json": {"()": JsonFormatter},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": formatter,
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"handlers": ["console"], "level": level},
            "loggers": {
                # These libraries log at INFO on every single request/model load,
                # which drowns out our own sync and routing events.
                "httpx": {"level": "WARNING"},
                "httpcore": {"level": "WARNING"},
                "chromadb": {"level": "WARNING"},
                "sentence_transformers": {"level": "WARNING"},
                "apscheduler": {"level": "WARNING"},
                "urllib3": {"level": "WARNING"},
                # Our own code should always be visible at the configured level.
                "app": {"level": level},
            },
        }
    )
