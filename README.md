# Darkcoal H3 Unsloth — RunPod Worker (Q4 Pruned Ref2VA)

Brand new repo built from scratch around **unsloth/MiniMax-H3-GGUF** — Q4 Pruned Ref2VA version.

- **DiT**: `minimax_h3_ref2va_pruned-Q4_K.gguf` (10.60 GiB) — for identity/appearance references【3263235363486924362†L170-L174】
- **Text Encoder**: `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (16.97 GiB) — Qwen3-VL-32B GGUF【3263235363486924362†L176-L180】, loaded with `CLIPLoaderGGUF`, type `minimax`【6441041265902894060†L5-L9】
- **Video VAE**: `minimax_h3_video_vae_fp16.safetensors` (5.21 GB) — required【3263235363486924362†L187-L190】
- **Audio VAE**: `minimax_h3_audio_vae_fp32.safetensors` (0.61 GB) — optional, for native stereo audio【3263235363486924362†L189-L192】
- **Tokenizer**: `vocab.json` + `merges.txt` from MiniMaxAI/MiniMax-H3 processor【3263235363486924362†L182-L186】

## Architecture

- Base: `nvidia/cuda:12.4.0-devel-ubuntu22.04` + Python 3.11 + ComfyUI 0.33.1
- Custom Nodes: `city96/ComfyUI-GGUF` (provides `UnetLoaderGGUF` and `CLIPLoaderGGUF`), ComfyUI native MiniMaxH3 nodes (core since 0.30)
- **GGUF loaders**: The repo uses `UnetLoaderGGUF` and `CLIPLoaderGGUF` in place of `UNETLoader` and `CLIPLoader` when loading from GGUF【911794564305999782†L44-L47】
- Serverless: RunPod Python SDK, handler starts ComfyUI API server on 8188 and forwards `runsync` requests
- Network Volume: `/runpod-volume/models/` (serverless) or `/workspace/models/` (Pods baking) — mapped via `extra_model_paths.yaml`

## Folder Structure

```
.
├── Dockerfile                # Builds GHCR image with ComfyUI + GGUF nodes + baked VAEs (optional)
├── extra_model_paths.yaml    # Maps unet_gguf, clip, vae to /runpod-volume/models/
├── handler.py                # RunPod serverless entrypoint
├── requirements.txt
├── workflows/
│   ├── t2v_api.json          # T2V test — no refs, Unsloth Q4 Ref2VA DiT in T2V mode
│   └── ref2va_q4_api.json    # Ref2VA — with <Picture 1> identity lock
├── scripts/
│   ├── bake_volume.py        # Python downloader for Pods -> /workspace
│   └── bake_volume.sh        # Shell wrapper for Pods command
├── .github/workflows/
│   └── build-and-push-ghcr.yml
└── playground/
    └── index.html            # Phone-first studio for new repo
```

## Quick Start — Brand New Repo

1. Create new GitHub repo (empty, no README)
2. Upload entire folder content to repo root
3. Enable GHCR: Repo Settings → Actions → General → Workflow permissions → Read and write
4. Push to `main` — GHCR workflow will build `ghcr.io/<you>/darkcoal-h3-unsloth:latest`
5. In RunPod → Serverless → New Template → Use `ghcr.io/<you>/darkcoal-h3-unsloth:latest`, set Env: `HF_TOKEN` (for gated downloads if needed)
6. Attach Network Volume (20GB+ free for models, but volume baking below will fill it)
7. Bake volume (see PODS command below)
8. Create Serverless Endpoint with that template + volume

## Volume Baking — RunPod PODS

See `scripts/bake_volume.sh` and `PODS_COMMAND.md`

In RunPod Pods (with volume attached at /workspace), run:

```bash
chmod +x /workspace/scripts/bake_volume.sh
/workspace/scripts/bake_volume.sh
```

This downloads into `/workspace/models/` which becomes `/runpod-volume/models/` in serverless.

## Playground

`playground/index.html` is a single-file app:
- Basic T2V default for testing
- Copyable logs (Copy / Select All)
- Editable model names
- Uses new repo's workflow (UnetLoaderGGUF + CLIPLoaderGGUF type minimax)

Open it locally or host on GitHub Pages.

## Model Files Reference

From Comfy-Org/MiniMax-H3 packaging:
- Text encoder `qwen3vl_32b_minimax_h3_*.safetensors` in `text_encoders`【3107504901050008902†L17-L20】, Video VAE `minimax_h3_video_vae_fp16.safetensors` in `vae`【3107504901050008902†L17-L20】, Audio VAE `minimax_h3_audio_vae_fp32.safetensors` in `vae`【3107504901050008902†L17-L20】

For GGUF variant, Unsloth repo provides:
- `minimax_h3_ref2va_pruned-Q4_K.gguf` for Ref2VA【3263235363486924362†L170-L174】
- `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` text encoder GGUF【3263235363486924362†L176-L180】

Workflow uses `CLIPLoaderGGUF` type `minimax` for that text encoder【6441041265902894060†L5-L9】 and `UnetLoaderGGUF` for DiT【911794564305999782†L44-L47】.
