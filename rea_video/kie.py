"""Kie generation: GPT Image 2 stills and Seedance 2.5 video.

Tasks are created at https://api.kie.ai/api/v1/jobs/createTask and read back from recordInfo.
Local files go through the file upload API first (kept about 24 hours):
https://docs.kie.ai/file-upload-api/quickstart

Docs used for the request shape:
https://docs.kie.ai/market/gpt/gpt-image-2-text-to-image
https://docs.kie.ai/market/gpt/gpt-image-2-image-to-image
https://docs.kie.ai/market/bytedance/seedance-2-5
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from . import __version__
from .errors import ProviderError, redact
from .net import encode_multipart, urllib_request
from .providers import (
    KIE_API_BASE,
    KIE_IMAGE_I2I,
    KIE_IMAGE_T2I,
    KIE_UPLOAD_BASE,
    KIE_VIDEO_MODEL,
    kie_image_aspect,
    kie_image_resolution,
    kie_poll_s,
    normalize_video_job,
    require_kie_key,
)


def _extension(mime: str) -> str:
    mime = (mime or "").lower()
    if "png" in mime:
        return "png"
    if "webp" in mime:
        return "webp"
    if "quicktime" in mime or mime.endswith("/mov"):
        return "mov"
    if "video" in mime or "mp4" in mime:
        return "mp4"
    if "jpeg" in mime or "jpg" in mime:
        return "jpg"
    return "bin"


def _mime_of(path: Path) -> str:
    name = path.suffix.lower()
    return {".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska"}.get(name, "application/octet-stream")


def classify_failure(fail_code, fail_msg) -> str:
    text = f"{fail_code or ''} {fail_msg or ''}".lower()
    if any(word in text for word in ("content_policy", "content policy", "safety", "nsfw", "moderation", "prohibited", "sensitive", "declined")):
        return "content_policy"
    if "invalid" in text and "image" in text or "aspect ratio" in text or "resolution too" in text:
        return "invalid_image"
    if "timeout" in text or "timed out" in text:
        return "generation_timeout"
    code = str(fail_code or "").strip()
    return code or "failed"


def kie_succeeded(res) -> bool:
    code = res.json.get("code") if isinstance(res.json, dict) else None
    if isinstance(code, int):
        return code == 200
    return 200 <= res.status < 300


def result_urls(info: dict) -> list[str]:
    raw = info.get("resultJson") if isinstance(info, dict) else None
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    urls = raw.get("resultUrls") or raw.get("result_urls") or []
    if not isinstance(urls, list):
        urls = [urls]
    out = []
    for item in urls:
        if isinstance(item, str) and item:
            out.append(item)
        elif isinstance(item, dict) and item.get("url"):
            out.append(str(item["url"]))
    return out


def image_body(*, prompt: str, aspect_ratio: str, resolution: str, image_urls: list[str] | None = None) -> dict:
    urls = [u for u in (image_urls or []) if u]
    payload = {
        "prompt": str(prompt or ""),
        "aspect_ratio": kie_image_aspect(aspect_ratio),
        "resolution": kie_image_resolution(resolution),
    }
    if urls:
        payload["input_urls"] = urls
    return {"model": KIE_IMAGE_I2I if urls else KIE_IMAGE_T2I, "input": payload}


def video_body(job: dict) -> dict:
    """Kie createTask body for one Seedance 2.5 call. ``job`` is the neutral form from normalize_video_job."""
    spec = normalize_video_job(job)
    payload = {
        "prompt": spec["prompt"],
        "duration": spec["seconds"],
        "resolution": spec["resolution"],
        "aspect_ratio": spec["aspect_ratio"],
        "generate_audio": spec["generate_audio"],
    }
    if spec["first_frame"]:
        payload["first_frame_url"] = spec["first_frame"]
    if spec["last_frame"]:
        payload["last_frame_url"] = spec["last_frame"]
    if spec["reference_images"]:
        payload["reference_image_urls"] = spec["reference_images"]
    if spec["reference_videos"]:
        payload["reference_video_urls"] = spec["reference_videos"]
    if spec["task_type"]:
        payload["omni_reference_task_type"] = spec["task_type"]
    return {"model": KIE_VIDEO_MODEL, "input": payload}


class KieClient:
    def __init__(self, key: str | None = None, transport=None, sleep=None, poll_s: float | None = None, timeout_s: float = 600.0):
        self.key = key or require_kie_key()
        self.transport = transport or urllib_request
        self.sleep = sleep or time.sleep
        self.poll_s = kie_poll_s() if poll_s is None else poll_s
        self.timeout_s = timeout_s

    def _failure(self, res, label: str) -> ProviderError:
        code = 0
        message = ""
        if isinstance(res.json, dict):
            try:
                code = int(res.json.get("code") or 0)
            except (TypeError, ValueError):
                code = 0
            message = redact(str(res.json.get("msg") or res.json.get("message") or ""), [self.key])
        status = code or res.status
        text = f"{label}: Kie {status}" + (f": {message}" if message else ": request failed")
        error_code = str((res.json or {}).get("code") or "") if isinstance(res.json, dict) else ""
        if status == 401 or res.status == 401:
            text = "Kie refused the API key. Check KIE_API_KEY in your .env file."
            error_code = "invalid_api_key"
        elif status == 402 or res.status == 402:
            text = "Your Kie balance doesn't cover this request. Add credit at https://kie.ai and run it again."
            error_code = "insufficient_credits"
        retry = 0.0
        try:
            retry = float(res.headers.get("retry-after") or 0)
        except (TypeError, ValueError):
            retry = 0.0
        err = ProviderError(text, status=status, code=error_code, retry_after=retry)
        err.retryable = status in (429, 500, 503) or res.status in (429, 500, 503)  # type: ignore[attr-defined]
        return err

    def _json(self, method: str, url: str, payload: dict | None = None, *, timeout: float, label: str, retries: int = 2) -> dict:
        headers = {"Authorization": f"Bearer {self.key}", "Accept": "application/json", "User-Agent": f"rea-video/{__version__}"}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        for attempt in range(retries + 1):
            res = self.transport(method, url, headers=headers, body=body, timeout=timeout)
            if kie_succeeded(res):
                return res.json or {}
            err = self._failure(res, label)
            if not getattr(err, "retryable", False) or attempt >= retries:
                raise err
            self.sleep(min(5.0, err.retry_after or self.poll_s or 0))
        raise ProviderError(f"{label}: Kie request failed.", code="failed")

    def prepare(self, source: str | Path, kind: str = "image") -> str:
        """A URL Kie can fetch. https URLs pass through. Local files are uploaded."""
        text = str(source)
        if text.startswith("https://") or text.startswith("http://"):
            if text.startswith("http://"):
                raise ProviderError(f"{kind}: Kie needs an https URL. Got an http URL.", code="upload_required")
            return text
        path = Path(text)
        if not path.is_file():
            raise ProviderError(f"{kind}: file not found: {path}", code="missing_file")
        mime = _mime_of(path)
        return self._upload(path.read_bytes(), mime, "videos/rea-video" if kind == "video" else "images/rea-video", path.name, label=kind)

    def _upload(self, data: bytes, mime: str, folder: str, filename: str, *, label: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in filename) or f"file.{_extension(mime)}"
        boundary = "rea-video-" + uuid.uuid4().hex
        body = encode_multipart([
            ("file", (safe, data, mime)),
            ("uploadPath", folder),
            ("fileName", safe),
        ], boundary)
        res = self.transport(
            "POST", f"{KIE_UPLOAD_BASE}/api/file-stream-upload",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": f"multipart/form-data; boundary={boundary}",
                     "User-Agent": f"rea-video/{__version__}"},
            body=body, timeout=180.0,
        )
        if not kie_succeeded(res) or (isinstance(res.json, dict) and res.json.get("success") is False):
            raise self._failure(res, label)
        data_obj = (res.json or {}).get("data") or {}
        url = data_obj.get("downloadUrl") or data_obj.get("fileUrl")
        if not url:
            raise ProviderError(f"{label}: Kie uploaded the file but returned no URL.", code="upload_failed")
        return str(url)

    def _wait(self, task_id: str, *, label: str) -> dict:
        started = time.monotonic()
        while True:
            info = self._record(task_id, label)
            state = str(info.get("state") or info.get("status") or "").lower()
            if state == "success":
                return info
            if state in ("fail", "failed"):
                message = redact(str(info.get("failMsg") or "Kie couldn't finish this task."), [self.key])
                raise ProviderError(f"{label}: {message}", code=classify_failure(info.get("failCode"), info.get("failMsg")))
            if time.monotonic() - started > self.timeout_s:
                raise ProviderError(f"{label}: Kie didn't finish within {self.timeout_s:.0f}s.", code="timeout")
            self.sleep(self.poll_s)

    def _record(self, task_id: str, label: str) -> dict:
        payload = self._json("GET", f"{KIE_API_BASE}/api/v1/jobs/recordInfo?taskId={task_id}", timeout=60, label=label)
        data = payload.get("data")
        return data if isinstance(data, dict) else payload

    def _create(self, body: dict, label: str) -> str:
        payload = self._json("POST", f"{KIE_API_BASE}/api/v1/jobs/createTask", body, timeout=300, label=label, retries=2)
        task_id = (payload.get("data") or {}).get("taskId") if isinstance(payload.get("data"), dict) else None
        if not task_id:
            raise ProviderError(f"{label}: Kie didn't return a task id.", code="missing_task")
        return str(task_id)

    def _download(self, url: str, label: str) -> bytes:
        res = self.transport("GET", url, headers={"User-Agent": f"rea-video/{__version__}"}, body=None, timeout=self.timeout_s)
        if res.status < 200 or res.status >= 300 or not res.body:
            raise ProviderError(f"{label}: couldn't download the result ({res.status}).", code="download_failed")
        return res.body

    def create_image(self, *, prompt: str, aspect_ratio: str, resolution: str = "1K",
                     image_urls: list[str] | None = None, label: str = "image") -> bytes:
        body = image_body(prompt=prompt, aspect_ratio=aspect_ratio, resolution=resolution, image_urls=image_urls)
        task_id = self._create(body, label)
        info = self._wait(task_id, label=label)
        url = (result_urls(info) or [None])[0]
        if not url:
            raise ProviderError(f"{label}: no image came back.", code="image_not_generated")
        return self._download(url, label)

    def create_video(self, job: dict, *, label: str = "video") -> bytes:
        body = video_body(job)
        task_id = self._create(body, label)
        info = self._wait(task_id, label=label)
        url = (result_urls(info) or [None])[0]
        if not url:
            raise ProviderError(f"{label}: Kie finished without a video URL.", code="missing_result")
        return self._download(url, label)
