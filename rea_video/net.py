"""JSON and binary HTTP. Callers decide what a status code means. Tests pass their own transport."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import __version__
from .errors import ProviderError


class HttpResult:
    def __init__(self, status: int, headers: dict, body: bytes):
        self.status = int(status or 0)
        self.headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        self.body = body or b""
        self.json = None
        if self.body:
            try:
                parsed = json.loads(self.body.decode("utf-8"))
            except (UnicodeError, ValueError):
                parsed = None
            self.json = parsed if isinstance(parsed, dict) else None


def urllib_request(method: str, url: str, *, headers: dict | None = None, body: bytes | None = None,
                   timeout: float = 120.0) -> HttpResult:
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={"User-Agent": f"rea-video/{__version__}", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return HttpResult(resp.status, dict(resp.headers), resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read() if exc.fp is not None else b""
        return HttpResult(exc.code, dict(exc.headers or {}), raw)
    except TimeoutError as exc:
        raise ProviderError(f"No answer from {url.split('/')[2]} within {timeout:.0f}s.", code="timeout") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if "timed out" in str(reason).lower():
            raise ProviderError(f"No answer from {url.split('/')[2]} within {timeout:.0f}s.", code="timeout") from exc
        raise ProviderError(f"Could not reach {url.split('/')[2]}: {reason}", code="network") from exc


def encode_multipart(fields: list[tuple[str, str | tuple[str, bytes, str]]], boundary: str) -> bytes:
    """`fields` values are text, or `(filename, bytes, mime)` for a file part."""
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.append(f"--{boundary}\r\n".encode())
        if isinstance(value, tuple):
            filename, data, mime = value
            chunks.append(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {mime}\r\n\r\n".encode()
            )
            chunks.append(data)
            chunks.append(b"\r\n")
        else:
            chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)
