import secrets
from typing import Optional

from fastapi import Header, HTTPException

from brain.settings import get_settings


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Optional API key check.

    If BRAIN_API_KEY is not set, every request is allowed (the original
    behavior). If it is set, requests must send a matching X-API-Key header.
    """
    expected = get_settings().brain_api_key
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(
        x_api_key, expected.get_secret_value()
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
