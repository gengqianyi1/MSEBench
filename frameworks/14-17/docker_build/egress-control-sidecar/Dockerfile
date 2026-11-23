# Harbor 0.21.0 egress-control sidecar (harbor/environments/docker/harbor-docker-egress-control-sidecar,
# Apache-2.0). Harbor starts it for every no-network trial on Linux and, when the image is missing,
# builds it itself FROM gogost/gost on Docker Hub. The other files in this directory are verbatim
# copies from the harbor 0.21.0 wheel; the only change here is that the base image can be overridden.
ARG GOST_IMAGE=gogost/gost:3.2.7-nightly.20260602@sha256:afc0137758ab4ce399d47a299f9abbacbf522b52a17e59cbb4b4e7a1a66e9196
FROM ${GOST_IMAGE}

COPY gost.yaml allowlist.txt /opt/egress-sidecar/
COPY --chmod=755 entrypoint.sh /opt/egress-sidecar/entrypoint.sh
COPY --chmod=755 bin/network-policy /usr/local/bin/network-policy
