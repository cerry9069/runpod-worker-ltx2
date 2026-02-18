"""
RunPod Serverless Handler — LTX Video Generation (v0.9.5)
Uses diffusers LTXConditionPipeline for text-to-video and image-to-video.

LTXConditionPipeline properly handles image conditioning with noise injection
(image_cond_noise_scale), which is critical for generating motion in I2V.
The older LTXImageToVideoPipeline has a known bug where it produces static
images due to missing noise on the first-frame latent.
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
CACHE_DIR = "/cache/ltx-v095"
MODEL_ID = "Lightricks/LTX-Video-0.9.5"


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


def load_pipeline():
    global PIPE

    os.makedirs(CACHE_DIR, exist_ok=True)
    clear_corrupt_cache()

    if PIPE is not None:
        return PIPE

    print(f"[ltx] Loading LTXConditionPipeline from {MODEL_ID}...")
    from diffusers import LTXConditionPipeline
    PIPE = LTXConditionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    ).to("cuda")
    PIPE.vae.enable_tiling()
    print("[ltx] Pipeline loaded.")
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
        pipe = load_pipeline()

        kwargs = {
            "prompt": prompt,
            "negative_prompt": inp.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"),
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "num_inference_steps": inp.get("steps", 40),
            "guidance_scale": inp.get("guidance_scale", 3.0),
            "decode_timestep": 0.05,
            "decode_noise_scale": 0.025,
            "max_sequence_length": inp.get("max_sequence_length", 256),
        }

        if image_url:
            from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
            image = download_image(image_url)
            image = image.resize((width, height))
            condition = LTXVideoCondition(image=image, frame_index=0)
            kwargs["conditions"] = [condition]
            # Noise injected into conditioning latents — critical for motion generation
            kwargs["image_cond_noise_scale"] = inp.get("image_cond_noise_scale", 0.15)

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
