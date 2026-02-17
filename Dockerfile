FROM runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2204

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    diffusers>=0.32.0 \
    transformers>=4.48.0 \
    accelerate \
    safetensors \
    sentencepiece>=0.2.0 \
    protobuf>=4.25.0 \
    huggingface_hub>=0.27.0 \
    pillow \
    requests \
    imageio[ffmpeg] \
    numpy

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
