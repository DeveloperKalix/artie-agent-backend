"""APScheduler factory for background jobs (news ingestion, etc.).

The scheduler itself is started/stopped from the FastAPI lifespan
context in app/main.py. Jobs are registered in later phases.
"""
from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler


def build_scheduler() -> AsyncIOScheduler:
    return AsyncIOScheduler(timezone="UTC")


__all__ = ["build_scheduler"]
