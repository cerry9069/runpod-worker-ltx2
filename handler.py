"""
RunPod Serverless Handler — LTX Video Generation
Uses diffusers LTXPipeline for text-to-video and image-to-video.
"""
import os
import runpod
import torch
import requests
import base64
import tempfile
from io import BytesIO
from PIL import Image

PIPE = None
I2V_PIPE = None
CACHE_DIR = "/cache/ltx-v091"
MODEL_ID = "Lightricks/LTX-Video-0.9.1"


def clear_corrupt_cache():
    """Check for corrupt spiece.model files and clear cache if found."""
    import shutil
    for root, dirs, files in os.walk(CACHE_DIR):
        for f in files:
            if f == "spiece.model":
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                if size < 10000:  # Real file is ~792KB; corrupt/LFS pointer is tiny
                    print(f"[ltx] WARNING: spiece.model is only {size} bytes, likely corrupt. Clearing cache.")
                    shutil.rmtree(CACHE_DIR, ignore_errors=True)
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    return


def load_pipeline(mode="t2v"):
    global PIPE, I2V_PIPE

    os.makedirs(CACHE_DIR, exist_ok=True)
    clear_corrupt_cache()

    if mode == "i2v":
        if I2V_PIPE is not None:
            return I2V_PIPE
        print(f"[ltx] Loading LTX Image-to-Video pipeline from {MODEL_ID}...")
        from diffusers import LTXImageToVideoPipeline
        I2V_PIPE = LTXImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        I2V_PIPE.enable_model_cpu_offload()
        print("[ltx] I2V pipeline loaded (cpu_offload).")
        return I2V_PIPE
    else:
        if PIPE is not None:
            return PIPE
        print(f"[ltx] Loading LTX Text-to-Video pipeline from {MODEL_ID}...")
        from diffusers import LTXPipeline
        PIPE = LTXPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        PIPE.enable_model_cpu_offload()
        print("[ltx] T2V pipeline loaded (cpu_offload).")
        return PIPE


def download_image(url):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def handler(event):
    inp = event.get("input", {})
    prompt = inp.get("prompt", "")

    if not prompt:
        return {"error": "prompt is required"}

    image_url = inp.get("image_url")
    duration = inp.get("duration", 5)
    width = inp.get("width", 768)
    height = inp.get("height", 512)
    fps = inp.get("fps", 24)
    num_frames = min(fps * duration, 97)  # Cap at 97 frames for VRAM safety

    try:
        kwargs = {
            "prompt": prompt,
            "negative_prompt": inp.get("negative_prompt", ""),
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "num_inference_steps": inp.get("steps", 50),
            "guidance_scale": inp.get("guidance_scale", 3.0),
            "max_sequence_length": inp.get("max_sequence_length", 256),
        }

        if image_url:
            pipe = load_pipeline("i2v")
            image = download_image(image_url)
            image = image.resize((width, height))
            kwargs["image"] = image
        else:
            pipe = load_pipeline("t2v")

        output = pipe(**kwargs)
        frames = output.frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        import imageio
        import numpy as np
        writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264")
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()

        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(tmp_path)

        return {
            "video_base64": video_b64,
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
        }
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
