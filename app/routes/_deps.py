"""Shared FastAPI dependencies for the route layer.

Keeping these in one place avoids sprinkling the same ``X-User-Id`` boilerplate
across every router.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException


def require_user_id(
    x_user_id: Optional[str] = Header(
        None,
        alias="X-User-Id",
        description="Caller's app user id (frontend auth subject).",
    ),
) -> str:
    """Extract and validate the caller's app user id.

    Every multi-tenant endpoint routes through this so the HTTP handlers
    themselves never need to know *how* we identify users — just that they
    receive a non-empty string.
    """
    uid = (x_user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=422, detail="Missing X-User-Id header")
    return uid


__all__ = ["require_user_id"]
