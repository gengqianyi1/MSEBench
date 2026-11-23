FROM python:3.11.9-slim
WORKDIR /workspace
COPY public/ /workspace/
RUN pip install --no-cache-dir pyyaml==6.0.2
CMD ["python", "-c", "import json, yaml; print('task20-public-ready')"]
