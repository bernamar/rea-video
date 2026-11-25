"""Which account reads a video, and which account generates stills and shots.

Mirrors the split used when both of your own keys are set:

- ``GEMINI_API_KEY`` (and no ``REA_VLM_*`` endpoint): analysis uses Gemini 3.1 Pro Preview.
- ``GEMINI_API_KEY`` and ``KIE_API_KEY`` together: remake plans default to Seedance 2.5 for
  video and GPT Image 2 for keyframes and stills, both on Kie. Media is uploaded through
  Kie's file API.
- Neither of those: the plan stays on MiniMax H3, and generation stays on Capsule Cinema.
- ``ZURELAY_API_KEY`` alone does not change the plan. ``rea-video generate`` can still send
  a Seedance 2.5 / GPT Image plan there when Kie is not configured.

A model you name, or ``rea-video prefs engine``, still wins over this default.
"""
from __future__ import annotations

import os

from .errors import ProviderError

# API model ids. Capability-map ids (seedance-2.5, gpt-image-2) are what plans store.
GEMINI_ANALYSIS_MODEL = "gemini-3.1-pro-preview"
KIE_API_BASE = "https://api.kie.ai"
KIE_UPLOAD_BASE = "https://kieai.redpandaai.co"
KIE_VIDEO_MODEL = "bytedance/seedance-2-5"
KIE_IMAGE_T2I = "gpt-image-2-text-to-image"
KIE_IMAGE_I2I = "gpt-image-2-image-to-image"
ZURELAY_BASE_URL = "https://api.zurelay.com/v1"
ZURELAY_VIDEO_MODEL = "seedance-2.5"
# Zuverse's Zurelay catalog lists GPT Image 2.5 Sunburst, not gpt-image-2.
ZURELAY_IMAGE_MODEL = "gpt-image-2.5-sunburst"

OWN_VIDEO_MODEL = "seedance-2.5"
OWN_IMAGE_MODEL = "gpt-image-2"
FALLBACK_VIDEO_MODEL = "minimax-h3"

GEMINI_KEY_URL = "https://aistudio.google.com/apikey"
KIE_KEY_URL = "https://kie.ai"
ZURELAY_KEY_URL = "https://zurelay.com"

# Kie Seedance 2.5 output. Reference clips are a tighter box; see reference_video_problem.
KIE_VIDEO_RESOLUTIONS = ("480p", "720p", "1080p", "4k")
KIE_IMAGE_RESOLUTIONS = ("1K", "2K", "4K")
# https://docs.kie.ai/market/bytedance/seedance-2-5
KIE_DURATION_MIN = 4
KIE_DURATION_MAX = 30
KIE_REF_VIDEO_MIN_S = 2
KIE_REF_VIDEO_MAX_S = 30
KIE_REF_VIDEO_TOTAL_S = 30
KIE_REF_VIDEOS = 10
KIE_REF_IMAGES = 30
KIE_IMAGE_INPUTS = 16
# Reference-video pixel box from the same page: product of width and height in [409600, 927408].
REF_PIXELS_MIN = 409_600
REF_PIXELS_MAX = 927_408
REF_SIDE_MIN = 300
REF_SIDE_MAX = 6000
REF_ASPECT_MIN = 0.4
REF_ASPECT_MAX = 2.5
REF_BYTES_MAX = 200 * 1024 * 1024
REF_FPS_MIN = 24
REF_FPS_MAX = 60

_IMAGE_RATIOS = (
    ("1:1", 1.0), ("3:2", 3 / 2), ("2:3", 2 / 3), ("4:3", 4 / 3), ("3:4", 3 / 4),
    ("5:4", 5 / 4), ("4:5", 4 / 5), ("16:9", 16 / 9), ("9:16", 9 / 16),
    ("2:1", 2.0), ("1:2", 0.5), ("3:1", 3.0), ("1:3", 1 / 3), ("21:9", 21 / 9), ("9:21", 9 / 21),
)
_VIDEO_RATIOS = (
    ("1:1", 1.0), ("4:3", 4 / 3), ("3:4", 3 / 4), ("16:9", 16 / 9), ("9:16", 9 / 16), ("21:9", 21 / 9),
)


def _clean(name: str) -> str:
    return str(os.getenv(name, "") or "").strip()


def gemini_key() -> str:
    return _clean("GEMINI_API_KEY") or _clean("GOOGLE_API_KEY")


def kie_key() -> str:
    return _clean("KIE_API_KEY")


def zurelay_key() -> str:
    return _clean("ZURELAY_API_KEY")


def use_gemini() -> bool:
    return bool(gemini_key())


def use_own_keys() -> bool:
    """Gemini reads, Kie generates. Both keys have to be set; one of them is not enough."""
    return bool(gemini_key() and kie_key())


def generation_backend() -> str | None:
    """Who ``rea-video generate`` calls. Kie wins. Zurelay is only the fallback for that command."""
    if kie_key():
        return "kie"
    if zurelay_key():
        return "zurelay"
    return None


def default_video_model() -> str:
    """Seedance 2.5 when both own keys are set, otherwise MiniMax H3."""
    return OWN_VIDEO_MODEL if use_own_keys() else FALLBACK_VIDEO_MODEL


def default_image_model() -> str | None:
    """GPT Image 2 when both own keys are set. Otherwise the plan does not name a stills model."""
    return OWN_IMAGE_MODEL if use_own_keys() else None


def analysis_model() -> str:
    """Gemini model id for ``GEMINI_API_KEY``. ``REA_GEMINI_MODEL`` overrides it."""
    return _clean("REA_GEMINI_MODEL") or GEMINI_ANALYSIS_MODEL


def require_kie_key() -> str:
    key = kie_key()
    if not key:
        raise ProviderError(
            f"No Kie API key. Create one at {KIE_KEY_URL} and set KIE_API_KEY in .env.",
            code="missing_api_key",
        )
    return key


def require_zurelay_key() -> str:
    key = zurelay_key()
    if not key:
        raise ProviderError(
            f"No Zurelay API key. Create one at {ZURELAY_KEY_URL} and set ZURELAY_API_KEY in .env.",
            code="missing_api_key",
        )
    return key


def kie_poll_s() -> float:
    try:
        n = float(os.getenv("REA_KIE_POLL_S", "2"))
    except (TypeError, ValueError):
        return 2.0
    return n if n >= 0 else 2.0


def _ratio_of(value: str) -> float | None:
    text = str(value or "").strip().lower().replace(" ", "")
    if not text or text in ("auto", "adaptive", "unknown"):
        return None
    sep = ":" if ":" in text else ("x" if "x" in text else "")
    if not sep:
        return None
    left, _, right = text.partition(sep)
    try:
        w, h = float(left), float(right)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return w / h


def _nearest(value: str, choices: tuple[tuple[str, float], ...], default: str) -> str:
    names = {name for name, _ in choices}
    text = str(value or "").strip()
    if text in names:
        return text
    ratio = _ratio_of(text)
    if ratio is None:
        return default
    best, delta = default, 1e9
    for name, target in choices:
        d = abs(ratio - target)
        if d < delta:
            best, delta = name, d
    return best


def kie_image_aspect(value: str) -> str:
    """Nearest GPT Image 2 aspect ratio. A bare ``9:16`` or ``1080x1920`` both work."""
    return _nearest(value, _IMAGE_RATIOS, "9:16")


def kie_video_aspect(value: str) -> str:
    """Nearest Seedance 2.5 aspect ratio. Unknown measurements stay ``adaptive`` (Kie's default)."""
    if not str(value or "").strip() or str(value).strip().lower() in ("unknown", "auto", "adaptive"):
        return "adaptive"
    return _nearest(value, _VIDEO_RATIOS, "adaptive")


def kie_image_resolution(value: str | None) -> str:
    text = str(value or "1K").strip().upper()
    if text in ("HIGH", "HD", "MAX"):
        return "4K"
    if text in KIE_IMAGE_RESOLUTIONS:
        return text
    return "1K"


def kie_video_resolution(value: str | None) -> str:
    text = str(value or "720p").strip().lower().replace("4K", "4k")
    if text == "4k":
        return "4k"
    if text in KIE_VIDEO_RESOLUTIONS:
        return text
    return "720p"


def clamp_duration(seconds: float) -> tuple[int, str]:
    """Kie Seedance 2.5 takes an integer from 4 to 30. Shorter plans are generated at 4s and trimmed."""
    try:
        n = int(round(float(seconds)))
    except (TypeError, ValueError):
        n = KIE_DURATION_MIN
    if n < KIE_DURATION_MIN:
        return KIE_DURATION_MIN, f"asked for {seconds:g}s; Kie minimum is {KIE_DURATION_MIN}s, so this call generates {KIE_DURATION_MIN}s and the extra is trimmed"
    if n > KIE_DURATION_MAX:
        return KIE_DURATION_MAX, f"asked for {seconds:g}s; Kie maximum is {KIE_DURATION_MAX}s, so this call generates {KIE_DURATION_MAX}s"
    return n, ""


def normalize_video_job(job: dict) -> dict:
    """One generation call in provider-neutral form, or ProviderError when Kie would reject it."""
    images = [str(u) for u in (job.get("reference_images") or []) if u]
    videos = [str(u) for u in (job.get("reference_videos") or []) if u]
    first = str(job.get("first_frame") or "") or None
    last = str(job.get("last_frame") or "") or None
    if last and not first:
        raise ProviderError(
            "Seedance 2.5 on Kie needs a first frame whenever a last frame is sent.",
            code="missing_first_frame",
        )
    if (images or videos) and (first or last):
        raise ProviderError(
            "Seedance 2.5 on Kie cannot take a first or last frame in the same request as reference images or video. "
            "Use first/last-frame mode, or reference mode.",
            code="exclusive_inputs",
        )
    if len(videos) > KIE_REF_VIDEOS:
        raise ProviderError(
            f"This call has {len(videos)} reference videos; Seedance 2.5 on Kie takes {KIE_REF_VIDEOS}.",
            code="too_many_videos",
        )
    if len(images) > KIE_REF_IMAGES:
        raise ProviderError(
            f"This call has {len(images)} reference images; Seedance 2.5 on Kie takes {KIE_REF_IMAGES}.",
            code="too_many_images",
        )
    try:
        ref_s = float(job.get("reference_seconds") or 0)
    except (TypeError, ValueError):
        ref_s = 0.0
    if videos and ref_s > KIE_REF_VIDEO_TOTAL_S + 1e-3:
        raise ProviderError(
            f"Reference video totals {ref_s:.1f}s. Seedance 2.5 on Kie allows {KIE_REF_VIDEO_TOTAL_S:.0f}s "
            "in one call (each clip 2–30s). Re-plan so the stretch is split.",
            code="reference_too_long",
        )
    seconds, note = clamp_duration(job.get("seconds") or KIE_DURATION_MIN)
    task_type = str(job.get("task_type") or "")
    if videos or images:
        # "reference" stops Kie treating words like "replace" as an edit, which would require duration -1.
        task_type = task_type or "reference"
    else:
        task_type = ""
    out = {
        "prompt": str(job.get("prompt") or ""),
        "seconds": seconds,
        "resolution": kie_video_resolution(job.get("resolution")),
        "aspect_ratio": kie_video_aspect(str(job.get("aspect_ratio") or "adaptive")),
        "generate_audio": job.get("generate_audio") is not False,
        "reference_images": images,
        "reference_videos": videos,
        "first_frame": first,
        "last_frame": last,
        "task_type": task_type,
    }
    if note:
        out["duration_note"] = note
    return out


def reference_video_problem(meta: dict) -> str:
    """Why a local clip cannot be a Kie reference video, or "" when it already fits.

    ``meta`` keys: seconds, width, height, fps, bytes. Missing measurements are not treated as failures.
    """
    try:
        seconds = float(meta.get("seconds") or 0)
    except (TypeError, ValueError):
        seconds = 0
    if seconds and (seconds < KIE_REF_VIDEO_MIN_S - 1e-3 or seconds > KIE_REF_VIDEO_MAX_S + 1e-3):
        return (f"reference video is {seconds:.1f}s; Kie Seedance 2.5 takes {KIE_REF_VIDEO_MIN_S:.0f}–"
                f"{KIE_REF_VIDEO_MAX_S:.0f}s per clip and {KIE_REF_VIDEO_TOTAL_S:.0f}s total")
    width, height = int(meta.get("width") or 0), int(meta.get("height") or 0)
    if width and height:
        aspect = width / height
        pixels = width * height
        if not (REF_ASPECT_MIN <= aspect <= REF_ASPECT_MAX):
            return f"reference video aspect {width}x{height} is outside {REF_ASPECT_MIN}–{REF_ASPECT_MAX}"
        if min(width, height) < REF_SIDE_MIN or max(width, height) > REF_SIDE_MAX or not (REF_PIXELS_MIN <= pixels <= REF_PIXELS_MAX):
            return (f"reference video is {width}x{height} ({pixels} px). Kie wants 480p or 720p, sides "
                    f"{REF_SIDE_MIN}–{REF_SIDE_MAX}px, and {REF_PIXELS_MIN}–{REF_PIXELS_MAX} total pixels")
    fps = meta.get("fps")
    if fps:
        try:
            fps_n = float(fps)
        except (TypeError, ValueError):
            fps_n = 0
        if fps_n and not (REF_FPS_MIN <= fps_n <= REF_FPS_MAX):
            return f"reference video is {fps_n:g} fps; Kie wants {REF_FPS_MIN}–{REF_FPS_MAX}"
    try:
        size = int(meta.get("bytes") or 0)
    except (TypeError, ValueError):
        size = 0
    if size > REF_BYTES_MAX:
        return f"reference video is {size / 1024 / 1024:.0f} MB; Kie allows {REF_BYTES_MAX // 1024 // 1024} MB"
    return ""


def fit_reference_size(width: int, height: int) -> tuple[int, int] | None:
    """Even size inside Kie's reference box, or None when ``width`` x ``height`` already fits.

    Returns None and leaves the caller to raise when the aspect itself is outside 0.4–2.5,
    because scaling cannot fix that.
    """
    if width <= 0 or height <= 0:
        return None
    if not reference_video_problem({"width": width, "height": height}):
        return None
    aspect = width / height
    if not (REF_ASPECT_MIN <= aspect <= REF_ASPECT_MAX):
        return None
    import math
    scale = min(1.0, math.sqrt(REF_PIXELS_MAX / (width * height)), REF_SIDE_MAX / max(width, height))
    # Prefer a short side of 720 when that still lands inside the pixel box (Kie's 720p reference cap).
    short = min(width, height)
    if short > 720:
        capped = min(scale, 720 / short)
        nw, nh = int(width * capped) // 2 * 2, int(height * capped) // 2 * 2
        if nw * nh >= REF_PIXELS_MIN and min(nw, nh) >= REF_SIDE_MIN:
            scale = capped
    nw, nh = max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)
    while nw * nh > REF_PIXELS_MAX and nw > 2 and nh > 2:
        if nw >= nh:
            nw -= 2
        else:
            nh -= 2
    if reference_video_problem({"width": nw, "height": nh}):
        return None
    return nw, nh


def stamp_image_tasks(tasks: list[dict], aspect_ratio: str) -> None:
    """Name GPT Image 2 on Kie for keyframes and stills when both own keys are set."""
    model = default_image_model()
    if not model or not tasks:
        return
    backend = generation_backend() or "kie"
    ratio = kie_image_aspect(aspect_ratio)
    for task in tasks:
        task["model"] = model
        task["provider"] = backend
        task["aspect_ratio"] = ratio


def top_up_url() -> str:
    if kie_key():
        return KIE_KEY_URL
    if zurelay_key():
        return "https://zurelay.com/app/billing"
    return KIE_KEY_URL
