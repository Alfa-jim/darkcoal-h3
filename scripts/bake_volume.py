#!/usr/bin/env python3
"""
RunPod PODS Volume Baker for Unsloth MiniMax-H3-GGUF Q4 Pruned Ref2VA
Downloads into /workspace/models/ which becomes /runpod-volume/models/ in serverless
"""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

BASE = Path("/workspace/models")
if not Path("/workspace").exists():
    BASE = Path("/runpod-volume/models")
    print(f"/workspace not found, using {BASE} (serverless mode)")

BASE.mkdir(parents=True, exist_ok=True)
(DIFF := BASE / "diffusion_models").mkdir(parents=True, exist_ok=True)
(TEXT := BASE / "text_encoders").mkdir(parents=True, exist_ok=True)
(VAE := BASE / "vae").mkdir(parents=True, exist_ok=True)
(PROC := TEXT / "processor").mkdir(parents=True, exist_ok=True)

print(f"Baking into {BASE}")

print("Downloading DiT: minimax_h3_ref2va_pruned-Q4_K.gguf (10.60 GiB)")
hf_hub_download(repo_id="unsloth/MiniMax-H3-GGUF", filename="minimax_h3_ref2va_pruned-Q4_K.gguf", local_dir=str(DIFF), local_dir_use_symlinks=False)

print("Downloading Text Encoder: qwen3vl_32b_minimax_h3-Q4_K_M.gguf (16.97 GiB)")
hf_hub_download(repo_id="unsloth/MiniMax-H3-GGUF", filename="qwen3vl_32b_minimax_h3-Q4_K_M.gguf", local_dir=str(TEXT), local_dir_use_symlinks=False)

print("Downloading Video VAE fp16 (5.21 GB)")
hf_hub_download(repo_id="Comfy-Org/MiniMax-H3", filename="vae/minimax_h3_video_vae_fp16.safetensors", local_dir=str(BASE), local_dir_use_symlinks=False)

print("Downloading Audio VAE fp32 (0.61 GB)")
hf_hub_download(repo_id="Comfy-Org/MiniMax-H3", filename="vae/minimax_h3_audio_vae_fp32.safetensors", local_dir=str(BASE), local_dir_use_symlinks=False)

print("Downloading vocab.json and merges.txt")
for fname in ["vocab.json", "merges.txt"]:
    hf_hub_download(repo_id="MiniMaxAI/MiniMax-H3", filename=f"processor/{fname}", local_dir=str(BASE), local_dir_use_symlinks=False)

# Move processor files to correct place
import shutil
for fname in ["vocab.json", "merges.txt"]:
    for p in BASE.rglob(fname):
        dst = PROC / fname
        if p != dst and "processor" in str(p):
            if not dst.exists():
                shutil.copy(str(p), str(dst))
        # also handle case where file landed at BASE/processor/fname
        alt = BASE / f"processor/{fname}"
        if alt.exists() and not dst.exists():
            shutil.move(str(alt), str(dst))

print("Done! Listing:")
for p in BASE.rglob("*"):
    if p.is_file():
        print(f"  {p.relative_to(BASE)} — {p.stat().st_size//(1024*1024)} MB")
