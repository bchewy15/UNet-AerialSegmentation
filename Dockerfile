# ill try a few base images, well see how slim works for now
FROM python:3-slim

WORKDIR /app

# not taking everything
COPY docker-reqs.txt .
COPY predict.py .
COPY model.py .
COPY saved_models/unetSeg.pt saved_models/unetSeg.pt

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r docker-reqs.txt

CMD ["python", "predict.py"]