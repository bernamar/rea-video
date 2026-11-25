"""Provider selection, Kie request shape, and cost estimates. No network."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rea_video.errors import ProviderError
from rea_video.gemini import DEFAULT_MODEL, GeminiClient, _http_message
from rea_video.generate import describe, reference_seconds
from rea_video.kie import KieClient, image_body, result_urls, video_body
from rea_video.net import HttpResult
from rea_video.pricing import estimate, image_usd, video_cost
from rea_video.providers import (
    GEMINI_ANALYSIS_MODEL,
    OWN_IMAGE_MODEL,
    OWN_VIDEO_MODEL,
    FALLBACK_VIDEO_MODEL,
    default_image_model,
    default_video_model,
    fit_reference_size,
    generation_backend,
    kie_image_aspect,
    normalize_video_job,
    reference_video_problem,
    stamp_image_tasks,
    use_own_keys,
)
from rea_video.vlm import make_client

GEMINI = "gemini-test-key-do-not-leak"
KIE = "kie-test-key-do-not-leak"


@pytest.fixture(autouse=True)
def _clean_keys(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "KIE_API_KEY", "ZURELAY_API_KEY", "REA_GEMINI_MODEL",
                 "REA_VLM_BASE_URL", "REA_VLM_API_KEY", "REA_VLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _own(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    monkeypatch.setenv("KIE_API_KEY", KIE)


def test_own_keys_need_both(monkeypatch):
    assert use_own_keys() is False
    assert default_video_model() == FALLBACK_VIDEO_MODEL
    assert default_image_model() is None
    assert generation_backend() is None
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    assert use_own_keys() is False
    assert default_video_model() == "minimax-h3"
    monkeypatch.setenv("KIE_API_KEY", KIE)
    assert use_own_keys() is True
    assert default_video_model() == OWN_VIDEO_MODEL
    assert default_image_model() == OWN_IMAGE_MODEL
    assert generation_backend() == "kie"


def test_zurelay_is_only_a_generate_fallback(monkeypatch):
    monkeypatch.setenv("ZURELAY_API_KEY", "zr_live_test")
    assert use_own_keys() is False
    assert default_video_model() == "minimax-h3"
    assert generation_backend() == "zurelay"
    monkeypatch.setenv("KIE_API_KEY", KIE)
    assert generation_backend() == "kie"


def test_analysis_defaults_to_gemini_31_pro_preview(monkeypatch):
    assert DEFAULT_MODEL == GEMINI_ANALYSIS_MODEL == "gemini-3.1-pro-preview"
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    assert GeminiClient().model == "gemini-3.1-pro-preview"
    monkeypatch.setenv("REA_GEMINI_MODEL", "gemini-3.7-flash")
    assert GeminiClient().model == "gemini-3.7-flash"


def test_explicit_vlm_endpoint_wins_over_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    monkeypatch.setenv("REA_VLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("REA_VLM_API_KEY", "doubao-key")
    monkeypatch.setenv("REA_VLM_MODEL", "doubao-seed-2.0-pro")
    client = make_client()
    assert client is not None
    assert "doubao-seed-2.0-pro" in client.describe()


def test_gemini_error_hides_the_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    message = _http_message(401, f"API key not valid: {GEMINI}")
    assert "refused" in message.lower() or "API key" in message
    assert GEMINI not in message
    assert "rate limit" in _http_message(429, "slow").lower()


def test_image_tasks_are_named_only_when_both_keys_are_set(monkeypatch):
    tasks = [{"shot": 1, "method": "t2i"}]
    stamp_image_tasks(tasks, "9:16")
    assert "model" not in tasks[0]
    _own(monkeypatch)
    stamp_image_tasks(tasks, "1080x1920")
    assert tasks[0]["model"] == "gpt-image-2"
    assert tasks[0]["provider"] == "kie"
    assert tasks[0]["aspect_ratio"] == "9:16"


def test_kie_aspect_and_reference_box():
    assert kie_image_aspect("864x1536") == "9:16"
    assert kie_image_aspect("1024x1024") == "1:1"
    assert reference_video_problem({"width": 720, "height": 1280, "seconds": 8, "fps": 30, "bytes": 1000}) == ""
    assert "720p" in reference_video_problem({"width": 1080, "height": 1920})
    assert fit_reference_size(1080, 1920) == (720, 1280)
    assert fit_reference_size(720, 1280) is None


def test_seedance_job_rejects_mixed_frame_and_reference_inputs():
    with pytest.raises(ProviderError) as exc:
        normalize_video_job({
            "prompt": "x", "seconds": 5, "first_frame": "https://example/a.png",
            "reference_videos": ["https://example/a.mp4"], "reference_seconds": 5,
        })
    assert exc.value.code == "exclusive_inputs"
    job = normalize_video_job({
        "prompt": "same shots", "seconds": 3, "resolution": "720p", "aspect_ratio": "9:16",
        "reference_images": ["https://example/face.png"],
        "reference_videos": ["https://example/clip.mp4"],
        "reference_seconds": 8,
        "task_type": "reference",
    })
    assert job["seconds"] == 4
    assert "minimum" in job["duration_note"]
    body = video_body(job)
    assert body["model"] == "bytedance/seedance-2-5"
    assert body["input"]["duration"] == 4
    assert body["input"]["reference_video_urls"] == ["https://example/clip.mp4"]
    assert body["input"]["reference_image_urls"] == ["https://example/face.png"]
    assert body["input"]["omni_reference_task_type"] == "reference"
    assert "first_frame_url" not in body["input"]


def test_reference_total_over_30s_is_refused():
    with pytest.raises(ProviderError) as exc:
        normalize_video_job({
            "prompt": "x", "seconds": 8, "reference_videos": ["https://example/a.mp4"], "reference_seconds": 31,
        })
    assert exc.value.code == "reference_too_long"


def test_image_body_picks_text_or_image_endpoint():
    text = image_body(prompt="a person", aspect_ratio="9:16", resolution="1K")
    assert text["model"] == "gpt-image-2-text-to-image"
    assert text["input"]["aspect_ratio"] == "9:16"
    assert text["input"]["resolution"] == "1K"
    assert "input_urls" not in text["input"]
    edit = image_body(prompt="redraw", aspect_ratio="16:9", resolution="2K", image_urls=["https://example/a.png"])
    assert edit["model"] == "gpt-image-2-image-to-image"
    assert edit["input"]["input_urls"] == ["https://example/a.png"]
    assert edit["input"]["resolution"] == "2K"


def _transport(handler):
    calls = []

    def transport(method, url, *, headers, body, timeout):
        call = {"method": method, "url": url, "headers": headers, "body": body}
        if isinstance(body, bytes) and body[:1] == b"{":
            call["json"] = json.loads(body.decode())
        calls.append(call)
        status, payload = handler(call)
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return HttpResult(status, {}, raw)

    return calls, transport


def test_kie_image_polls_and_downloads(monkeypatch):
    polls = {"n": 0}

    def handler(call):
        if "createTask" in call["url"]:
            return 200, {"code": 200, "data": {"taskId": "task_img"}}
        if "recordInfo" in call["url"]:
            polls["n"] += 1
            if polls["n"] == 1:
                return 200, {"code": 200, "data": {"state": "generating"}}
            return 200, {"code": 200, "data": {"state": "success", "resultJson": {"resultUrls": ["https://cdn.example/out.png"]}}}
        assert call["url"] == "https://cdn.example/out.png"
        return 200, b"\x89PNG"

    calls, transport = _transport(handler)
    client = KieClient(key=KIE, transport=transport, sleep=lambda _s: None, poll_s=0)
    assert client.create_image(prompt="a person", aspect_ratio="864x1536", resolution="1K", label="casting") == b"\x89PNG"
    created = next(call for call in calls if "createTask" in call["url"])
    assert created["json"]["model"] == "gpt-image-2-text-to-image"
    assert created["json"]["input"]["aspect_ratio"] == "9:16"
    assert created["headers"]["Authorization"] == f"Bearer {KIE}"
    assert result_urls({"resultJson": {"resultUrls": ["https://cdn.example/out.png"]}}) == ["https://cdn.example/out.png"]


def test_kie_credit_error_hides_the_key():
    def handler(_call):
        return 200, {"code": 402, "msg": f"balance empty for {KIE}"}

    _calls, transport = _transport(handler)
    client = KieClient(key=KIE, transport=transport, sleep=lambda _s: None, poll_s=0)
    with pytest.raises(ProviderError) as exc:
        client.create_image(prompt="x", aspect_ratio="1:1", label="casting")
    assert exc.value.status == 402
    assert exc.value.code == "insufficient_credits"
    assert KIE not in str(exc.value)


def test_kie_upload_uses_the_file_api(tmp_path):
    image = tmp_path / "face.png"
    image.write_bytes(b"\x89PNG")

    def handler(call):
        assert "file-stream-upload" in call["url"]
        assert b'filename="face.png"' in call["body"]
        assert call["headers"]["Authorization"] == f"Bearer {KIE}"
        return 200, {"code": 200, "success": True, "data": {"downloadUrl": "https://temp.example/face.png"}}

    _calls, transport = _transport(handler)
    client = KieClient(key=KIE, transport=transport, sleep=lambda _s: None, poll_s=0)
    assert client.prepare(image, "image") == "https://temp.example/face.png"
    assert client.prepare("https://example.com/already.png", "image") == "https://example.com/already.png"


def test_prices_follow_kie_and_zurelay():
    assert video_cost("kie", seconds=10, resolution="720p") == {"output": 3.15, "reference": 0, "total": 3.15}
    assert video_cost("kie", seconds=15, resolution="720p", reference_seconds=10) == {"output": 2.85, "reference": 1.9, "total": 4.75}
    assert video_cost("kie", seconds=10, resolution="480p", reference_seconds=10) == {"output": 0.85, "reference": 0.85, "total": 1.7}
    assert video_cost("zurelay", seconds=10, resolution="720p")["total"] == 3.05
    assert image_usd("kie", "1K") == 0.03
    assert image_usd("kie", "4K") == 0.08
    assert image_usd("zurelay") == 0.034
    got = estimate(provider="kie", output_seconds=5, reference_seconds=0, images=1, resolution="720p", image_resolution="1K")
    assert got["as_of"] == "2026-11-25"
    assert got["read_usd"] == 0.07


def test_describe_prices_a_seedance_plan_and_refuses_other_models(monkeypatch):
    _own(monkeypatch)
    plan = {
        "model": "seedance-2.5",
        "units": [{
            "id": "G1", "strategy": "remix", "call_s": 8, "calls": 1, "shots": [1], "draft": "follow the clip",
            "refs": [{"tag": "@视频1", "ranges": [[0, 8]]}, {"tag": "@图片1"}],
        }],
        "image_tasks": [{"shot": 1, "method": "t2i", "prompt": "frame"}],
        "prompts": {"G1": "final prompt"},
        "spec": {"api_params": {"omni_reference_task_type": "reference"}},
    }
    preview = describe(plan, aspect_ratio="9:16")
    assert preview["provider"] == "kie"
    assert preview["video_api_model"] == "bytedance/seedance-2-5"
    assert preview["image_model"] == "gpt-image-2"
    assert preview["units"][0]["seconds"] == 8
    assert preview["units"][0]["reference_seconds"] == 8
    assert reference_seconds(plan["units"][0]) == 8
    assert preview["cost"]["total"] == estimate(
        provider="kie", output_seconds=8, reference_seconds=8, images=1, resolution="720p", image_resolution="1K",
    )["total"]
    plan["model"] = "minimax-h3"
    with pytest.raises(ProviderError) as exc:
        describe(plan)
    assert exc.value.code == "model_not_supported"


def test_cli_dry_run_prints_estimate_without_calling(monkeypatch, tmp_path, capsys):
    _own(monkeypatch)
    out = tmp_path / "ref"
    out.mkdir()
    plan = {
        "model": "seedance-2.5",
        "units": [{"id": "G1", "strategy": "text", "call_s": 5, "calls": 1, "shots": [1], "refs": [], "draft": "a street"}],
        "image_tasks": [],
        "prompts": {},
        "spec": {},
    }
    (out / "breakdown.json").write_text(json.dumps({"source": {"aspect_ratio": "9:16"}, "production_plan": plan}), encoding="utf-8")
    from rea_video.cli import main

    assert main(["generate", str(out), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "bytedance/seedance-2-5" in printed
    assert "G1" in printed


def test_fit_reference_video_scales_a_tall_clip(tmp_path):
    src = tmp_path / "tall.mp4"
    made = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30", "-t", "3", "-pix_fmt", "yuv420p", str(src)],
        capture_output=True, check=False,
    )
    if made.returncode != 0 or not src.is_file():
        pytest.skip("ffmpeg could not make a test clip")
    from rea_video.generate import fit_reference_video, probe_video

    fitted = fit_reference_video(src, tmp_path / "fit.mp4")
    meta = probe_video(fitted)
    assert reference_video_problem({k: meta[k] for k in ("width", "height", "fps", "seconds", "bytes")}) == ""
    assert meta["width"] == 720
    assert Path(fitted) != src
