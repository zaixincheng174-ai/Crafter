FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/crafter-sam3/hf \
    PORT=8765 PORT_HEALTH=8765
RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
RUN git clone --depth 1 https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && test "$(git -C /opt/sam3 rev-parse HEAD)" = f1647e8d0c1d280991f4d1b7d974c2704537fe66 \
    && pip install -e /opt/sam3 flask portalocker einops 'opencv-python-headless<4.12' pycocotools scipy psutil \
    && git clone --depth 1 https://github.com/HaozheZhao/Crafter.git /opt/crafter \
    && test "$(git -C /opt/crafter rev-parse HEAD)" = 6398457f33179a45ff581f7d73efdcf92be2cd82
COPY prefetch.py serve.py start.sh /opt/service/
RUN python -m py_compile /opt/service/prefetch.py /opt/service/serve.py \
    && python -c 'from sam3.model_builder import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor'
EXPOSE 8765
CMD ["bash", "/opt/service/start.sh"]
