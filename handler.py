import os
import sys
import json
import time
import base64
import shutil
import traceback
from pathlib import Path
import threading
import subprocess

import runpod
from huggingface_hub import hf_hub_download

COMFYUI_PATH = Path("/comfyui")
INPUT_DIR = COMFYUI_PATH / "input"
OUTPUT_DIR = COMFYUI_PATH / "output"
MODELS_BASE = Path("/runpod-volume/models") if Path("/runpod-volume/models").exists() else COMFYUI_PATH / "models"

# Ensure dirs
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def check_models():
    """Check required Unsloth Q4 models"""
    required = {
        "diffusion_models/minimax_h3_ref2va_pruned-Q4_K.gguf": MODELS_BASE / "diffusion_models/minimax_h3_ref2va_pruned-Q4_K.gguf",
        "text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf": MODELS_BASE / "text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        "vae/minimax_h3_video_vae_fp16.safetensors": MODELS_BASE / "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors": MODELS_BASE / "vae/minimax_h3_audio_vae_fp32.safetensors",
        "text_encoders/processor/vocab.json": MODELS_BASE / "text_encoders/processor/vocab.json",
        "text_encoders/processor/merges.txt": MODELS_BASE / "text_encoders/processor/merges.txt",
    }
    missing = []
    for key, path in required.items():
        # also check fallback in /comfyui/models
        fallback = COMFYUI_PATH / "models" / key
        if not path.exists() and not fallback.exists():
            missing.append(key)
    if missing:
        print(f"[Darkcoal-Unsloth] MISSING MODELS: {missing}", flush=True)
        print(f"[Darkcoal-Unsloth] MODELS_BASE: {MODELS_BASE}, listing: {list(MODELS_BASE.rglob('*.gguf'))[:10]}", flush=True)
    else:
        print(f"[Darkcoal-Unsloth] All Unsloth Q4 models found at {MODELS_BASE}", flush=True)
    return missing

def start_comfyui_server():
    """Start ComfyUI API server in background thread"""
    def run_server():
        cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"]
        print(f"[ComfyUI] Starting: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(COMFYUI_PATH))
        proc.wait()
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    # wait for server
    import requests
    for i in range(60):
        try:
            r = requests.get("http://127.0.0.1:8188/system_stats", timeout=2)
            if r.status_code == 200:
                print(f"[ComfyUI] Server ready after {i}s", flush=True)
                return True
        except:
            pass
        time.sleep(1)
    print("[ComfyUI] Server failed to start", flush=True)
    return False

COMFY_READY = start_comfyui_server()
check_models()

def save_images(images):
    """images: list of {name, image: dataUrl} -> save to input dir, return list of filenames"""
    saved = []
    for img in images:
        name = img.get("name", f"input_{int(time.time())}.png")
        data = img.get("image", "")
        if "," in data:
            data = data.split(",",1)[1]
        try:
            b = base64.b64decode(data)
            path = INPUT_DIR / name
            path.write_bytes(b)
            saved.append(name)
        except Exception as e:
            print(f"Failed to save image {name}: {e}")
    return saved

def build_workflow(base_workflow, prompt, width, height, duration, steps, seed, ref_images):
    wf = json.loads(json.dumps(base_workflow))  # deep copy
    length = duration * 24
    for nid, node in wf.items():
        inputs = node.get("inputs", {})
        if "prompt" in inputs:
            inputs["prompt"] = prompt
        if "width" in inputs:
            inputs["width"] = width
        if "height" in inputs:
            inputs["height"] = height
        if "length" in inputs:
            inputs["length"] = length
        if "steps" in inputs:
            inputs["steps"] = steps
        if "seed" in inputs or "noise_seed" in inputs:
            if "seed" in inputs:
                inputs["seed"] = seed
            if "noise_seed" in inputs:
                inputs["noise_seed"] = seed
    # Handle ref images if provided
    # For Unsloth Ref2VA, ref_images are list of [node_id, 0]
    # We create LoadImage nodes for each ref if not already present
    if ref_images:
        # Find main node
        main_id = None
        for nid, node in wf.items():
            if node.get("class_type") == "MiniMaxH3ReferenceToVideo":
                main_id = nid
                break
        if main_id:
            wf[main_id]["inputs"]["ref_images"] = []
            for idx, img_name in enumerate(ref_images):
                load_id = str(100+idx)
                wf[load_id] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
                wf[main_id]["inputs"]["ref_images"].append([load_id, 0])
    return wf

def run_comfyui_workflow(workflow):
    import requests
    # Queue prompt
    payload = {"prompt": workflow}
    try:
        r = requests.post("http://127.0.0.1:8188/prompt", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        prompt_id = data.get("prompt_id")
        print(f"Queued prompt {prompt_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to queue workflow: {e} {traceback.format_exc()}")

    # Poll history
    for _ in range(600):  # 10 min max
        try:
            r = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=10)
            if r.status_code == 200:
                hist = r.json()
                if prompt_id in hist:
                    status = hist[prompt_id].get("status", {})
                    if status.get("completed") or hist[prompt_id].get("outputs"):
                        outputs = hist[prompt_id].get("outputs", {})
                        # Find SaveVideo output
                        for node_id, out in outputs.items():
                            if "gifs" in out or "images" in out or "videos" in out:
                                # Return first video
                                videos = out.get("videos") or out.get("gifs") or out.get("images") or []
                                if videos:
                                    # ComfyUI saves to output dir, we need to read file
                                    for v in videos:
                                        fname = v.get("filename")
                                        fpath = OUTPUT_DIR / fname
                                        if fpath.exists():
                                            b64 = base64.b64encode(fpath.read_bytes()).decode()
                                            return f"data:video/mp4;base64,{b64}", out
                        # Fallback: check output dir latest file
                        files = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
                        if files:
                            b64 = base64.b64encode(files[0].read_bytes()).decode()
                            return f"data:video/mp4;base64,{b64}", outputs
                        return None, outputs
        except Exception as e:
            print(f"Poll error: {e}")
        time.sleep(2)
    raise TimeoutError("ComfyUI workflow timed out")

def handler(job):
    try:
        job_input = job.get("input", {})
        # Workflow can be passed directly or use default t2v
        workflow = job_input.get("workflow")
        if not workflow:
            # Load default T2V workflow
            default_path = COMFYUI_PATH / "workflows" / "t2v_api.json"
            if not default_path.exists():
                default_path = Path("/workflows/t2v_api.json")
            workflow = json.loads(default_path.read_text())

        prompt = job_input.get("prompt", workflow.get("10", {}).get("inputs", {}).get("prompt", "A young woman in cozy bedroom, photoreal"))
        width = job_input.get("width", 720)
        height = job_input.get("height", 1280)
        duration = job_input.get("duration", 6)
        steps = job_input.get("steps", 24)
        seed = job_input.get("seed", int(time.time()) % 4294967295)

        images = job_input.get("images", [])
        saved_names = save_images(images) if images else []

        final_wf = build_workflow(workflow, prompt, width, height, duration, steps, seed, saved_names)

        if not COMFY_READY:
            return {"error": "ComfyUI server not ready, models missing", "missing": check_models()}

        video_b64, outputs = run_comfyui_workflow(final_wf)
        if video_b64:
            return {"status": "COMPLETED", "output": {"video": video_b64, "images": [{"data": video_b64}], "prompt_id": "done"}}
        else:
            return {"status": "FAILED", "error": "No video output", "outputs": str(outputs)[:2000]}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Handler error: {e}\n{tb}", flush=True)
        return {"status": "FAILED", "error": str(e), "traceback": tb[:5000]}

runpod.serverless.start({"handler": handler})
