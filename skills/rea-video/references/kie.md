# Kie 上生成复刻镜头

`GEMINI_API_KEY` 和 `KIE_API_KEY` 都设置时，没指定模型也没有 `prefs engine`，复刻方案默认是 Seedance 2.5，关键帧和静帧是 GPT Image 2，都走 Kie。只设了其中一个，或者两个都没设，方案仍是 MiniMax H3（或你保存的引擎），生成仍走 capsule-cinema。

分析跟生成是分开的：只设 `GEMINI_API_KEY`、又没有 `REA_VLM_*` 时，拉片用 Gemini 3.1 Pro Preview（`REA_GEMINI_MODEL` 可以改）。看一条大约 30 秒的片子，粗算 0.07 美元。

## 费用

Kie 标价，2026-11-25 核对。失败的任务不扣费。

- Seedance 2.5，没有参考视频：480p 0.14 美元/秒，720p 0.315，1080p 0.79，4K 1.255。
- 有参考视频时改用较低单价，乘（参考视频总秒数 + 输出秒数）：480p 0.085，720p 0.19，1080p 0.475，4K 0.775。参考图和参考音频不计秒。
- GPT Image 2：1K 0.03 美元/张，2K 0.05，4K 0.08。

## 怎么跑

```bash
rea-video generate out/ref --dry-run
rea-video generate out/ref --yes
```

`--dry-run` 不调用。不加 `--yes` 也只显示估算。成片在 `out/ref/remake/generate/`。一次生成多个镜头时，保存的是一整段，不按镜头切开。

参考视频按 Kie 的限制：单段 2–30 秒，一次合计不超过 30 秒，480p 或 720p（总像素 409600–927408），24–60 fps，200 MB 以内，mp4 / mov / mkv。超出画面尺寸的片段会先用 ffmpeg 缩小。比 2 秒更短或比 30 秒更长的片段不会在这里裁，需要重新规划。

首帧 / 首尾帧不能和参考图、参考视频放在同一次请求里。

没有 `KIE_API_KEY`、但有 `ZURELAY_API_KEY` 时，`generate` 把同一条 Seedance 方案改走 Zurelay。Zurelay 的静帧模型是它目录里的 GPT Image 2.5 Sunburst，不是 GPT Image 2。这不会把默认方案从 MiniMax H3 改掉。
