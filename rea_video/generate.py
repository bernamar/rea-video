"""Run a Seedance 2.5 remake on Kie, with Zurelay as the fallback.

``describe`` only reads the plan and prices it. ``execute`` uploads media and creates tasks.
Neither one changes the breakdown. Capsule Cinema remains the path for every other model,
including MiniMax H3 when the Kie and Zurelay keys are absent.

Reference clips that are larger than Kie's 480p/720p box are scaled with ffmpeg before upload.
A clip shorter than 2s or longer than 30s is refused: the plan has to be split, not trimmed here.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .errors import ProviderError
from .kie import KieClient
from .pricing import estimate
from .providers import (
    KIE_IMAGE_I2I,
    KIE_IMAGE_T2I,
    OWN_IMAGE_MODEL,
    OWN_VIDEO_MODEL,
    ZURELAY_IMAGE_MODEL,
    ZURELAY_VIDEO_MODEL,
    fit_reference_size,
    generation_backend,
    kie_image_aspect,
    kie_image_resolution,
    kie_video_aspect,
    kie_video_resolution,
    normalize_video_job,
    reference_video_problem,
)
from .util import LOG
from .zurelay import ZurelayClient


def reference_seconds(unit: dict) -> float:
    total = 0.0
    for ref in unit.get("refs") or []:
        if str(ref.get("tag") or "").startswith("@视频"):
            for span in ref.get("ranges") or []:
                if len(span) >= 2:
                    total += max(0.0, float(span[1]) - float(span[0]))
    return total


def _image_api_model(provider: str, has_inputs: bool) -> str:
    if provider == "zurelay":
        return ZURELAY_IMAGE_MODEL
    return KIE_IMAGE_I2I if has_inputs else KIE_IMAGE_T2I


def describe(plan: dict, *, provider: str | None = None, resolution: str = "720p",
             image_resolution: str = "1K", aspect_ratio: str = "9:16", only: str | None = None) -> dict:
    """What generate would send, and what it would cost. No files are read and nothing is billed."""
    model = plan.get("model") or ""
    if model != OWN_VIDEO_MODEL:
        raise ProviderError(
            f"This plan targets {model or 'no model'}. rea-video generate runs Seedance 2.5 "
            f"({OWN_VIDEO_MODEL}) and GPT Image 2 stills. Set both GEMINI_API_KEY and KIE_API_KEY "
            f"and re-plan, or run `rea-video plan <dir> --model {OWN_VIDEO_MODEL}`. "
            "Other models, including MiniMax H3, stay on `rea-video remake-run` (Capsule Cinema).",
            code="model_not_supported",
        )
    backend = provider or generation_backend()
    if not backend:
        raise ProviderError(
            "No generation key. Set KIE_API_KEY to generate on Kie (with GEMINI_API_KEY, new plans "
            "default to Seedance 2.5 and GPT Image 2). Or set ZURELAY_API_KEY to run that same "
            "Seedance path on Zurelay. MiniMax H3 still uses `rea-video remake-run`.",
            code="missing_api_key",
        )
    resolution = kie_video_resolution(resolution)
    image_resolution = kie_image_resolution(image_resolution)
    aspect = kie_video_aspect(aspect_ratio)
    units = [u for u in (plan.get("units") or []) if not only or u.get("id") == only]
    if only and not units:
        raise ProviderError(f"No generation unit {only} in this plan.", code="missing_unit")
    tasks = [t for t in (plan.get("image_tasks") or []) if not only or _task_in_units(t, units)]
    notes: list[str] = []
    described = []
    for unit in units:
        frames = unit.get("strategy") == "frames"
        job = normalize_video_job({
            "prompt": (plan.get("prompts") or {}).get(unit.get("id")) or unit.get("draft") or "",
            "seconds": (unit.get("call_s") or unit.get("generate_s") or unit.get("duration") or 4),
            "resolution": resolution,
            "aspect_ratio": aspect,
            "reference_seconds": reference_seconds(unit) if unit.get("strategy") in ("remix", "edit") else 0,
            # Placeholders exist only so the exclusive-input check sees which kinds this unit sends.
            "reference_images": [] if frames else ["placeholder"] * _count(unit, "@图片"),
            "reference_videos": [] if frames else ["placeholder"] * _count(unit, "@视频"),
            "first_frame": "placeholder" if frames else None,
            "task_type": ((plan.get("spec") or {}).get("api_params") or {}).get("omni_reference_task_type") or "",
        })
        if job.get("duration_note"):
            notes.append(f"{unit.get('id')}: {job['duration_note']}")
        described.append({
            "id": unit.get("id"),
            "strategy": unit.get("strategy"),
            "seconds": job["seconds"],
            "reference_seconds": round(reference_seconds(unit), 3) if unit.get("strategy") in ("remix", "edit") else 0,
            "images": _count(unit, "@图片"),
            "videos": _count(unit, "@视频"),
            "note": job.get("duration_note") or "",
        })
    calls_of = {u.get("id"): max(1, int(u.get("calls") or 1)) for u in units}
    ref_total = sum(item["reference_seconds"] for item in described)
    out_total = sum(item["seconds"] * calls_of.get(item["id"], 1) for item in described)
    cost = estimate(
        provider=backend,
        output_seconds=out_total,
        reference_seconds=ref_total,
        images=len(tasks),
        resolution=resolution,
        image_resolution=image_resolution,
    )
    image_model = OWN_IMAGE_MODEL if backend == "kie" else ZURELAY_IMAGE_MODEL
    if backend == "zurelay":
        notes.append("Zurelay stills use gpt-image-2.5-sunburst. Kie stills use gpt-image-2. Video is Seedance 2.5 on either.")
        if resolution == "4k":
            notes.append("Zurelay has no 4k Seedance rate on the 2026-10-15 card; the call is sent at 1080p.")
    return {
        "provider": backend,
        "video_model": OWN_VIDEO_MODEL,
        "video_api_model": "bytedance/seedance-2-5" if backend == "kie" else ZURELAY_VIDEO_MODEL,
        "image_model": image_model,
        "image_api_model": _image_api_model(backend, any(t.get("method") == "i2i" for t in tasks)),
        "resolution": resolution,
        "image_resolution": image_resolution,
        "aspect_ratio": aspect,
        "image_aspect_ratio": kie_image_aspect(aspect_ratio),
        "units": described,
        "image_count": len(tasks),
        "cost": cost,
        "notes": notes,
    }


def attach_cost(plan: dict, aspect_ratio: str = "") -> None:
    """Record a Kie or Zurelay estimate on a Seedance 2.5 plan when a generation key is set."""
    if plan.get("model") != OWN_VIDEO_MODEL or not (plan.get("units") or plan.get("image_tasks")):
        return
    backend = generation_backend()
    if not backend:
        return
    try:
        preview = describe(plan, provider=backend, aspect_ratio=aspect_ratio or "9:16")
    except ProviderError:
        return
    plan["provider"] = preview["provider"]
    plan["image_model"] = preview["image_model"]
    plan["cost"] = preview["cost"]


def _count(unit: dict, prefix: str) -> int:
    return sum(1 for ref in unit.get("refs") or [] if str(ref.get("tag") or "").startswith(prefix))


def _task_in_units(task: dict, units: list[dict]) -> bool:
    shots = {s for unit in units for s in (unit.get("shots") or [])}
    return task.get("shot") in shots


def client_for(provider: str | None = None, **kwargs):
    backend = provider or generation_backend()
    if backend == "kie":
        return KieClient(**kwargs)
    if backend == "zurelay":
        return ZurelayClient(**kwargs)
    raise ProviderError(
        "No generation key. Set KIE_API_KEY, or ZURELAY_API_KEY as the fallback.",
        code="missing_api_key",
    )


def probe_video(path: Path) -> dict:
    """width, height, fps, seconds. Empty dict when ffprobe cannot read the file."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,avg_frame_rate,duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        data = json.loads(out.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    fps = 0.0
    rate = str(stream.get("avg_frame_rate") or "")
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            fps = float(num) / float(den) if float(den) else 0.0
        except ValueError:
            fps = 0.0
    try:
        seconds = float(stream.get("duration") or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        "seconds": seconds,
        "bytes": path.stat().st_size if path.is_file() else 0,
    }


def fit_reference_video(path: Path, dest: Path) -> Path:
    """Return ``path`` when it already fits Kie's reference-video box, otherwise a scaled copy."""
    meta = probe_video(path)
    problem = reference_video_problem(meta)
    if not problem:
        return path
    if meta.get("seconds") and (meta["seconds"] < 2 or meta["seconds"] > 30):
        raise ProviderError(f"{path.name}: {problem}", code="bad_reference")
    fitted = fit_reference_size(int(meta.get("width") or 0), int(meta.get("height") or 0))
    fps_bad = meta.get("fps") and not (24 <= float(meta["fps"]) <= 60)
    if fitted is None and not fps_bad:
        raise ProviderError(f"{path.name}: {problem}", code="bad_reference")
    dest.parent.mkdir(parents=True, exist_ok=True)
    width, height = fitted or (int(meta.get("width") or 0), int(meta.get("height") or 0))
    cmd = ["ffmpeg", "-y", "-i", str(path), "-an"]
    if fitted:
        cmd += ["-vf", f"scale={width}:{height}:flags=lanczos"]
    if fps_bad:
        cmd += ["-r", "30"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300, check=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ProviderError(f"Could not scale {path.name} to Kie's reference-video limits ({problem}).", code="bad_reference") from exc
    LOG.info("scaled reference %s to %dx%d for Kie", path.name, width, height)
    return dest


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def execute(plan: dict, out_dir: Path, preview: dict, *, client, breakdown: dict | None = None,
            images: bool = True, videos: bool = True) -> dict:
    """Upload inputs and write outputs under ``out_dir/remake/generate``. ``client`` is a Kie or Zurelay client."""
    out_dir = Path(out_dir)
    root = out_dir / "remake" / "generate"
    root.mkdir(parents=True, exist_ok=True)
    written: dict = {"provider": preview["provider"], "images": {}, "videos": {}}
    if images:
        previous: list[Path] = []
        wanted = {item["id"] for item in preview["units"]}
        shots = {s for unit in plan.get("units") or [] if unit.get("id") in wanted for s in (unit.get("shots") or [])}
        for task in plan.get("image_tasks") or []:
            if shots and task.get("shot") not in shots:
                continue
            inputs = _image_files(task, out_dir, plan.get("intent") or {}, previous)
            urls = [client.prepare(p, "image") for p in inputs]
            data = client.create_image(
                prompt=task.get("prompt") or "",
                aspect_ratio=task.get("aspect_ratio") or preview["image_aspect_ratio"],
                resolution=preview["image_resolution"],
                image_urls=urls,
                label=f"shot {task.get('shot')} {task.get('role') or 'still'}",
            )
            target = out_dir / str(task.get("target") or f"remake/redrawn/shot{int(task.get('shot') or 0):03d}.png")
            _write(target, data)
            previous.append(target)
            written["images"][f"{task.get('shot')}:{task.get('role') or 'still'}"] = str(target)
    if videos:
        from .integrations.capsule_cinema import ref_clips

        b = breakdown or {}
        wanted = {item["id"] for item in preview["units"]}
        for unit in plan.get("units") or []:
            if unit.get("id") not in wanted:
                continue
            clips = ref_clips(b, out_dir, unit) if b else {}
            images_u, videos_u, first, last = _unit_media(unit, clips)
            for ref in unit.get("refs") or []:
                raw = ref.get("path")
                if raw and str(ref.get("tag") or "").startswith("@图片"):
                    path = Path(raw)
                    if path.is_file() and path not in images_u:
                        images_u.append(path)
            video_urls = []
            for i, path in enumerate(videos_u, 1):
                fitted = fit_reference_video(path, root / f"{unit['id']}_ref{i}.mp4")
                meta = probe_video(fitted)
                problem = reference_video_problem(meta)
                if problem:
                    raise ProviderError(f"{unit['id']}: {problem}", code="bad_reference")
                video_urls.append(client.prepare(fitted, "video"))
            image_urls = [client.prepare(p, "image") for p in images_u]
            first_url = client.prepare(first, "image") if first else None
            last_url = client.prepare(last, "image") if last else None
            prompt = (plan.get("prompts") or {}).get(unit["id"]) or unit.get("draft") or ""
            data = client.create_video({
                "prompt": prompt,
                "seconds": next(item["seconds"] for item in preview["units"] if item["id"] == unit["id"]),
                "resolution": preview["resolution"],
                "aspect_ratio": preview["aspect_ratio"],
                "reference_images": image_urls,
                "reference_videos": video_urls,
                "reference_seconds": next(item["reference_seconds"] for item in preview["units"] if item["id"] == unit["id"]),
                "first_frame": first_url,
                "last_frame": last_url,
                "task_type": ((plan.get("spec") or {}).get("api_params") or {}).get("omni_reference_task_type") or "",
            }, label=unit["id"])
            dest = _write(root / f"{unit['id']}.mp4", data)
            written["videos"][unit["id"]] = {"path": str(dest), "shots": list(unit.get("shots") or [])}
    (root / "results.json").write_text(json.dumps(written, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return written


def _image_files(task: dict, out_dir: Path, intent: dict, previous: list[Path]) -> list[Path]:
    if task.get("method") != "i2i":
        return []
    found: list[Path] = []
    src = task.get("source_frame") or ""
    if src:
        path = Path(src)
        if not path.is_file():
            path = out_dir / src
        if path.is_file():
            found.append(path)
    for swap in intent.get("swaps") or []:
        image = swap.get("image")
        if image and Path(image).is_file():
            found.append(Path(image))
    if previous:
        found.append(previous[-1])
    # Kie image-to-image takes 16 inputs.
    uniq, seen = [], set()
    for path in found:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(path)
    return uniq[:16]


def _unit_media(unit: dict, clips: dict) -> tuple[list[Path], list[Path], Path | None, Path | None]:
    images, videos = [], []
    first = last = None
    for (part, tag), raw in clips.items():
        path = Path(raw)
        if tag == "first" and part == 1:
            first = path
        elif tag == "last" and part == 1:
            last = path
        elif str(tag).startswith("@视频"):
            videos.append(path)
        elif str(tag).startswith("@图片"):
            images.append(path)
    return images, videos, first, last
