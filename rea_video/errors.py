"""Errors from a paid provider. The message is safe to print; API keys are never included."""
from __future__ import annotations

import os


class ProviderError(RuntimeError):
    """A generation or analysis call that the user can act on. `code` is a short machine reason."""

    def __init__(self, message: str, *, status: int = 0, code: str = "", retry_after: float = 0.0):
        super().__init__(message)
        self.status = int(status or 0)
        self.code = code or ""
        self.retry_after = float(retry_after or 0)


def secret_values() -> list[str]:
    """Every key this process might have loaded. Used only to strip them out of error text."""
    names = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "KIE_API_KEY", "ZURELAY_API_KEY", "REA_VLM_API_KEY")
    return [os.getenv(name, "") or "" for name in names]


def redact(message: str, secrets: list[str] | None = None) -> str:
    keys = list(secret_values())
    keys.extend(secrets or [])
    out = str(message or "")
    for key in keys:
        trimmed = str(key or "").strip()
        if len(trimmed) > 6:
            out = out.replace(trimmed, "…")
    return out[:500]
