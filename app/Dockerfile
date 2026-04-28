FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    ultralytics \
    sahi \
    tqdm \
    pyyaml \
    shapely \
    pycocotools \
    rasterio \
    opencv-python-headless \
    pillow \
    "numpy<2.0"

RUN pip install --no-cache-dir \
    rfdetr==1.6.5 \
    transformers>=5.0.0

RUN pip install --no-cache-dir \
    "streamlit==1.40.0" \
    opencv-python-headless \
    pillow

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]