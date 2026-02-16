FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    ltx-pipelines \
    transformers \
    accelerate \
    safetensors \
    pillow \
    requests \
    imageio[ffmpeg] \
    numpy

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
