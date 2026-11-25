"""Zurelay fallback for ``rea-video generate`` when KIE_API_KEY is not set.

Same Seedance 2.5 video id Zuverse sends (``seedance-2.5``). Stills use
``gpt-image-2.5-sunburst``, the image model Zuverse's Zurelay catalog lists.
Files go inline as data URLs (Zurelay's limit is about 20 MB a file). Larger
reference clips need Kie, which uploads them instead.

https://zurelay.com
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from . import __version__
from .errors import ProviderError, redact
from .net import urllib_request
from .util import LOG
from .providers import (
    ZURELAY_BASE_URL,
    ZURELAY_IMAGE_MODEL,
    ZURELAY_VIDEO_MODEL,
    normalize_video_job,
    require_zurelay_key,
)

INLINE_MAX = 18 * 1024 * 1024


def _mime_of(path: Path) -> str:
    return {".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".mp4": "video/mp4", ".mov": "video/quicktime"}.get(path.suffix.lower(), "application/octet-stream")


class ZurelayClient:
    def __init__(self, key: str | None = None, transport=None, base_url: str | None = None, sleep=None, poll_s: float = 2.0):
        self.key = key or require_zurelay_key()
        self.transport = transport or urllib_request
        self.base = (base_url or ZURELAY_BASE_URL).rstrip("/")
        self.sleep = sleep or time.sleep
        self.poll_s = poll_s

    def _failure(self, res, label: str) -> ProviderError:
        err = (res.json or {}).get("error") if isinstance(res.json, dict) else None
        code = str((err or {}).get("code") or (err or {}).get("type") or "")
        message = redact(str((err or {}).get("message") or ""), [self.key])
        if res.status == 401:
            text, code = "Zurelay refused the API key. Check ZURELAY_API_KEY in your .env file.", "invalid_api_key"
        elif res.status == 402:
            text, code = "Your Zurelay balance doesn't cover this request. Add credit at https://zurelay.com/app/billing and run it again.", "insufficient_credits"
        elif res.status == 413:
            text = "A file is too large for Zurelay (about 20 MB a file). Use a shorter clip, or set KIE_API_KEY so the file is uploaded instead."
            code = code or "too_large"
        else:
            text = f"{label}: Zurelay {res.status}" + (f" {code}: {message}" if message or code else ": request failed")
        return ProviderError(text, status=res.status, code=code)

    def _json(self, method: str, path: str, payload: dict | None, *, timeout: float, label: str) -> dict:
        headers = {"Authorization": f"Bearer {self.key}", "Accept": "application/json", "User-Agent": f"rea-video/{__version__}"}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        res = self.transport(method, self.base + path, headers=headers, body=body, timeout=timeout)
        if 200 <= res.status < 300 and isinstance(res.json, dict):
            return res.json
        raise self._failure(res, label)

    def prepare(self, source: str | Path, kind: str = "image") -> str:
        text = str(source)
        if text.startswith("https://") or text.startswith("data:"):
            return text
        path = Path(text)
        if not path.is_file():
            raise ProviderError(f"{kind}: file not found: {path}", code="missing_file")
        size = path.stat().st_size
        if size > INLINE_MAX:
            raise ProviderError(
                f"{kind}: {path.name} is {size / 1024 / 1024:.1f} MB. Zurelay takes about 20 MB inline. "
                "Set KIE_API_KEY to upload it instead.",
                code="too_large",
            )
        mime = _mime_of(path)
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()

    def create_image(self, *, prompt: str, aspect_ratio: str, resolution: str = "1K",
                     image_urls: list[str] | None = None, label: str = "image") -> bytes:
        urls = [u for u in (image_urls or []) if u]
        body = {"model": ZURELAY_IMAGE_MODEL, "prompt": str(prompt or ""), "n": 1, "response_format": "url"}
        if aspect_ratio:
            sizes = {"9:16": "864x1536", "16:9": "1536x864", "1:1": "1024x1024", "3:4": "1024x1536", "4:3": "1536x1024", "21:9": "1536x640"}
            body["size"] = sizes.get(str(aspect_ratio), aspect_ratio)
        path = "/images/edits" if urls else "/images/generations"
        if urls:
            body["images"] = urls
        payload = self._json("POST", path, body, timeout=600, label=label)
        item = (payload.get("data") or [None])[0] or {}
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"])
        if item.get("url"):
            res = self.transport("GET", item["url"], headers={"User-Agent": f"rea-video/{__version__}"}, body=None, timeout=120)
            if res.status < 200 or res.status >= 300 or not res.body:
                raise ProviderError(f"{label}: couldn't download the image ({res.status}).", code="download_failed")
            return res.body
        raise ProviderError(f"{label}: no image came back.", code="image_not_generated")

    def create_video(self, job: dict, *, label: str = "video") -> bytes:
        spec = normalize_video_job(job)
        resolution = spec["resolution"]
        if resolution == "4k":
            LOG.warning("Zurelay's Seedance card has no 4k rate; sending 1080p")
            resolution = "1080p"
        body = {
            "model": ZURELAY_VIDEO_MODEL,
            "prompt": spec["prompt"],
            "seconds": spec["seconds"],
            "resolution": resolution,
            "aspect_ratio": spec["aspect_ratio"],
            "generate_audio": spec["generate_audio"],
        }
        if spec["reference_images"]:
            body["reference_images"] = spec["reference_images"]
        if spec["reference_videos"]:
            body["reference_videos"] = spec["reference_videos"]
        if spec["first_frame"]:
            body["first_frame"] = spec["first_frame"]
        if spec["last_frame"]:
            body["last_frame"] = spec["last_frame"]
        created = self._json("POST", "/videos", body, timeout=300, label=label)
        task_id = created.get("id")
        if not task_id:
            raise ProviderError(f"{label}: Zurelay didn't return a video id.", code="missing_task")
        # One status read. The caller (generate) polls by calling create_video, which waits here.
        return self._download_video(str(task_id), label)

    def _download_video(self, task_id: str, label: str) -> bytes:
        deadline = time.monotonic() + 600
        while True:
            info = self._json("GET", f"/videos/{task_id}", None, timeout=60, label=label)
            status = str(info.get("status") or "").lower()
            if status in ("completed", "succeeded", "success"):
                res = self.transport("GET", f"{self.base}/videos/{task_id}/content",
                                     headers={"Authorization": f"Bearer {self.key}", "User-Agent": f"rea-video/{__version__}"},
                                     body=None, timeout=600)
                if 200 <= res.status < 300 and res.body:
                    return res.body
                raise ProviderError(f"{label}: couldn't download the video ({res.status}).", code="download_failed")
            if status in ("failed", "error", "cancelled", "canceled"):
                err = info.get("error") or {}
                message = redact(str(err.get("message") or status), [self.key])
                raise ProviderError(f"{label}: {message}", code=str(err.get("code") or "failed"))
            if time.monotonic() > deadline:
                raise ProviderError(f"{label}: Zurelay didn't finish within 600s.", code="timeout")
            self.sleep(self.poll_s)
