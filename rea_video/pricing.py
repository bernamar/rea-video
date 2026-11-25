"""What a remake costs, in US dollars.

Kie prices checked 2026-11-25 against https://kie.ai/seedance-2-5 and https://kie.ai/gpt-image-2.
Failed Kie tasks are not charged (https://kie.ai/pricing).

Zurelay prices are the ones Zuverse recorded from https://zurelay.com/models on 2026-10-15,
used only when ``rea-video generate`` falls back to Zurelay. Failed Zurelay videos are free there too.

Gemini 3.1 Pro Preview, standard tier, is about $2 / 1M input tokens and $12 / 1M output tokens
for prompts up to 200k (https://ai.google.dev/gemini-api/docs/pricing). A 30s watch is roughly
300 tokens a second plus a few thousand output tokens, about $0.07. That is a reading estimate,
not part of the remake total.
"""
from __future__ import annotations

KIE_PRICES_AS_OF = "2026-11-25"
ZURELAY_PRICES_AS_OF = "2026-10-15"

# Seedance 2.5 on Kie, USD per second of output when no reference video is sent.
KIE_SEEDANCE_PER_S = {"480p": 0.14, "720p": 0.315, "1080p": 0.79, "4k": 1.255}
# With a reference video the lower rate covers uploaded seconds and output seconds.
KIE_SEEDANCE_WITH_VIDEO_PER_S = {"480p": 0.085, "720p": 0.19, "1080p": 0.475, "4k": 0.775}
# GPT Image 2 on Kie, USD per image.
KIE_IMAGE = {"1K": 0.03, "2K": 0.05, "4K": 0.08}

# Seedance 2.5 on Zurelay, per second of output. 4K is not on the 2026-10-15 card.
ZURELAY_SEEDANCE_PER_S = {"480p": 0.135, "720p": 0.305, "1080p": 0.75}
ZURELAY_SEEDANCE_REFERENCE_PER_S = {"480p": 0.081, "720p": 0.183, "1080p": 0.183}
ZURELAY_IMAGE = 0.034

GEMINI_READ_USD = 0.07  # one ~30s watch on Gemini 3.1 Pro Preview, roughly


def prices_as_of(provider: str) -> str:
    return ZURELAY_PRICES_AS_OF if provider == "zurelay" else KIE_PRICES_AS_OF


def image_usd(provider: str, resolution: str = "1K") -> float:
    if provider == "zurelay":
        return ZURELAY_IMAGE
    return KIE_IMAGE.get(resolution, KIE_IMAGE["1K"])


def video_cost(provider: str, *, seconds: float, resolution: str, reference_seconds: float = 0) -> dict:
    """Output seconds, plus reference-clip seconds when a reference video is sent."""
    res = str(resolution or "720p").lower()
    has_ref = reference_seconds > 0
    if provider == "zurelay":
        output_rate = ZURELAY_SEEDANCE_PER_S.get(res, ZURELAY_SEEDANCE_PER_S["720p"])
        ref_rate = ZURELAY_SEEDANCE_REFERENCE_PER_S.get(res, ZURELAY_SEEDANCE_REFERENCE_PER_S["720p"])
        out = output_rate * seconds
        ref = ref_rate * reference_seconds if has_ref else 0.0
    else:
        output_table = KIE_SEEDANCE_PER_S
        with_video = KIE_SEEDANCE_WITH_VIDEO_PER_S
        # On Kie the with-video rate replaces the output rate and also covers the clip.
        rate = (with_video if has_ref else output_table).get(res) or (with_video if has_ref else output_table)["720p"]
        ref_rate = with_video.get(res, with_video["720p"])
        out = rate * seconds
        ref = ref_rate * reference_seconds if has_ref else 0.0
    return {"output": round2(out), "reference": round2(ref), "total": round2(out + ref)}


def estimate(*, provider: str, output_seconds: float, reference_seconds: float, images: int,
             resolution: str = "720p", image_resolution: str = "1K") -> dict:
    video = video_cost(provider, seconds=output_seconds, resolution=resolution, reference_seconds=reference_seconds)
    stills = round2(image_usd(provider, image_resolution) * max(0, images))
    return {
        "provider": provider,
        "as_of": prices_as_of(provider),
        "resolution": resolution,
        "image_resolution": image_resolution,
        "output_seconds": round2(output_seconds),
        "reference_seconds": round2(reference_seconds),
        "images": images,
        "video": video["total"],
        "video_output": video["output"],
        "video_reference": video["reference"],
        "image_usd": stills,
        "total": round2(video["total"] + stills),
        "read_usd": GEMINI_READ_USD,
    }


def round2(n: float) -> float:
    return round(float(n) * 100) / 100
