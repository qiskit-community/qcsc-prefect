"""Fugaku-specific Prefect blocks."""

from qcsc_prefect_blocks.fugaku.blocks import (
    CommandBlock,
    ExecutionProfileBlock,
    HPCProfileBlock,
)

__all__ = [
    "CommandBlock",
    "ExecutionProfileBlock",
    "HPCProfileBlock",
]
