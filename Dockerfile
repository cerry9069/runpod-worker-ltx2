FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    pillow \
    requests \
    imageio[ffmpeg] \
    numpy

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
