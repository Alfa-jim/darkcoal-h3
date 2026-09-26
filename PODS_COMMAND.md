# RunPod PODS Volume Baking Command

This repo uses /workspace as volume mount in Pods, which becomes /runpod-volume in Serverless.

## One-liner to bake volume in a RunPod Pod

1. Create a RunPod Pod with Network Volume (50GB+ recommended, same region as serverless endpoint)
2. Inside Pod terminal:

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/darkcoal-h3-unsloth.git /workspace/repo
cd /workspace/repo
chmod +x scripts/bake_volume.sh
./scripts/bake_volume.sh
```

What it downloads into /workspace/models/:
- diffusion_models/minimax_h3_ref2va_pruned-Q4_K.gguf (10.60 GiB) Ref2VA Q4
- text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf (16.97 GiB) text encoder
- vae/minimax_h3_video_vae_fp16.safetensors (5.21 GB) video VAE
- vae/minimax_h3_audio_vae_fp32.safetensors (0.61 GB) audio VAE
- text_encoders/processor/vocab.json + merges.txt tokenizer

Verify:
```bash
ls -lh /workspace/models/diffusion_models/
ls -lh /workspace/models/text_encoders/
ls -lh /workspace/models/vae/
du -sh /workspace/models/*
```

Then attach same Network Volume to Serverless Endpoint using ghcr.io/<you>/darkcoal-h3-unsloth:latest
