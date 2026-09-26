FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV COMFYUI_PATH=/comfyui

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git wget curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && pip install --upgrade pip

# ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui && \
    cd /comfyui && pip install -r requirements.txt

# Custom Nodes - ComfyUI-GGUF (provides UnetLoaderGGUF and CLIPLoaderGGUF)
# GGUF needs a loader that core does not have: install city96/ComfyUI-GGUF and use its UNet / CLIP loaders【911794564305999782†L44-L47】
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    cd ComfyUI-GGUF && pip install -r requirements.txt

# Optional: Qwen3VL TE fix for GGUF shape errors (recommended for qwen3vl_32b gguf)
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/pottokao-dotcom/ComfyUI-GGUF-Qwen3VL-TE.git || true

# ComfyUI Manager (helps debug)
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git || true

WORKDIR /comfyui

# Copy repo files
COPY requirements.txt /comfyui/requirements.txt
RUN pip install -r /comfyui/requirements.txt

COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY workflows/ /comfyui/workflows/
COPY handler.py /handler.py
COPY scripts/ /scripts/

# Bake lightweight VAEs into image as fallback (heavy GGUFs stay on network volume)
# From Comfy-Org/MiniMax-H3: video VAE fp16 (5.21 GB)【3263235363486924362†L187-L190】 and audio VAE fp32 (0.61 GB)【3263235363486924362†L189-L192】
RUN mkdir -p /comfyui/models/vae && \
    wget -q -O /comfyui/models/vae/minimax_h3_video_vae_fp16.safetensors \
    https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors && \
    wget -q -O /comfyui/models/vae/minimax_h3_audio_vae_fp32.safetensors \
    https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors

# Processor tokenizer for Qwen3VL GGUF (required)【3263235363486924362†L182-L186】
RUN mkdir -p /comfyui/models/text_encoders/processor && \
    wget -q -O /comfyui/models/text_encoders/processor/vocab.json \
    https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/processor/vocab.json && \
    wget -q -O /comfyui/models/text_encoders/processor/merges.txt \
    https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/processor/merges.txt

ENV PYTHONPATH=/comfyui

# RunPod handler
CMD ["python", "/handler.py"]
