"""
RunPod Serverless Handler — LTX-2 Video Generation
Uses the official ltx-pipelines package from Lightricks.
Supports both text-to-video and image-to-video.
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
CACHE_DIR = "/cache/ltx2"


def load_pipeline():
    global PIPE
    if PIPE is not None:
        return PIPE

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("[ltx2] Loading LTX-2 pipeline...")

    from ltx_pipelines import LTXVideoPipeline

    PIPE = LTXVideoPipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    ).to("cuda")

    print("[ltx2] Pipeline loaded.")
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
    num_frames = min(fps * duration, 257)  # LTX-2 max ~257 frames

    try:
        pipe = load_pipeline()

        kwargs = {
            "prompt": prompt,
            "negative_prompt": inp.get("negative_prompt", ""),
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "num_inference_steps": inp.get("steps", 30),
            "guidance_scale": inp.get("guidance_scale", 7.0),
        }

        if image_url:
            image = download_image(image_url)
            image = image.resize((width, height))
            kwargs["image"] = image

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
