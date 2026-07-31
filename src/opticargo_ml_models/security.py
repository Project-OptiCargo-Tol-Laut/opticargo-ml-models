from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from .config import get_settings


async def require_internal_token(
    x_internal_service_token: str | None = Header(default=None),
) -> None:
    settings = get_settings()
    expected = settings.internal_service_token
    if not expected and settings.opticargo_environment == "development":
        return
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="INTERNAL_SERVICE_TOKEN belum dikonfigurasi.",
        )
    if not x_internal_service_token or not secrets.compare_digest(x_internal_service_token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token internal tidak valid.")
