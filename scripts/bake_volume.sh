#!/bin/bash
set -e
echo "=== Darkcoal H3 Unsloth Volume Baker ==="
echo "Base: /workspace/models (will be /runpod-volume/models in serverless)"
pip install -q huggingface_hub
mkdir -p /workspace/models/diffusion_models /workspace/models/text_encoders /workspace/models/vae /workspace/models/text_encoders/processor
python3 /workspace/scripts/bake_volume.py || python3 ./scripts/bake_volume.py || python3 /scripts/bake_volume.py
echo "=== Bake complete ==="
ls -lh /workspace/models/diffusion_models/ || true
ls -lh /workspace/models/text_encoders/ || true
ls -lh /workspace/models/vae/ || true
