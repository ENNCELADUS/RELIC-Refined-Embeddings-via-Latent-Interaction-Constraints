"""Distributed helpers for DDP-capable orchestration."""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def initialize_distributed(ddp_enabled: bool) -> None:
    """Initialize distributed context.

    Args:
        ddp_enabled: Whether distributed mode is enabled in config.

    This scaffold keeps initialization lightweight and no-op by default.
    """
    if ddp_enabled:
        LOGGER.info("DDP requested; using single-process scaffold mode.")
