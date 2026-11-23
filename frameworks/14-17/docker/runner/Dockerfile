# Harbor runner for tasks 14-17. It ships in docker/ so the organizer host needs no PyPI, no Docker Hub
# and no download.docker.com: the host only needs a Docker daemon. Built by docker/build_images.sh, whose
# build context holds just the package requirements.txt.
FROM python:3.12-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg unzip \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" \
      > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin docker-buildx-plugin \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/chembench/requirements.txt
RUN pip install --no-cache-dir -r /opt/chembench/requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1
CMD ["bash"]
