# Private response sidecar for ChemBench tasks 16 and 17.
#
# WHAT IS IN THIS IMAGE, AND WHY IT MUST STAY PRIVATE
# ---------------------------------------------------
# It contains the task's `tests/task_snapshot/`, which includes `private_data/` -- the enumerated
# reference optimum, the score-band thresholds and the naive-policy ceilings. Anyone who can pull
# this image can read the answer. It must never be pushed to a public registry, and it must never
# be used as the agent environment's base image.
#
# BUILD CONTEXT
# -------------
# The server is a FRAMEWORK asset (`templates/sidecar/private_response_server.py`, shared by both
# tasks and containing no model coefficients); the data is a TASK asset. They live in different
# subtrees, so the build context is the PACKAGE ROOT:
#
#     docker build -f sidecar/Dockerfile \
#       --build-arg TASK_DIR=tasks/task-16-tablet-microenvironment-cqa \
#       --build-arg TASK_NUMBER=16 \
#       -t chembench/task16-sidecar:2.1.0-r10 .
#
# Use `sidecar/build_sidecar.sh`, which does this and prints the digest to freeze.
FROM mambaorg/micromamba:2.8.1

ARG TASK_DIR
ARG TASK_NUMBER
ARG PORT=8716

# The sidecar solves the same scientific model as the verifier, so it pins the verifier's lock file
# rather than a second, independently drifting dependency set.
COPY --chown=$MAMBA_USER:$MAMBA_USER ${TASK_DIR}/tests/environment.lock.yml /tmp/environment.lock.yml
RUN micromamba create -y -n chembench -f /tmp/environment.lock.yml && micromamba clean --all --yes
ENV PATH=/opt/conda/envs/chembench/bin:$PATH
ENV MAMBA_DEFAULT_ENV=chembench

USER root
RUN mkdir -p /snapshot /srv && chown -R $MAMBA_USER:$MAMBA_USER /snapshot /srv
USER $MAMBA_USER

# The organizer snapshot: private_data/ (model + reference) and input/ (public config the model
# already has). The server reads both; nothing here is ever mounted into the agent container.
COPY --chown=$MAMBA_USER:$MAMBA_USER ${TASK_DIR}/tests/task_snapshot/ /snapshot/
COPY --chown=$MAMBA_USER:$MAMBA_USER templates/sidecar/private_response_server.py /srv/private_response_server.py

ENV CHEMBENCH_SIDECAR_TASK_NUMBER=${TASK_NUMBER}
ENV CHEMBENCH_SIDECAR_PORT=${PORT}
EXPOSE ${PORT}

# --host 0.0.0.0 rather than the server's 127.0.0.1 default: with `network_mode: service:main` the
# sidecar shares the agent container's network namespace, so 127.0.0.1 would also work, but binding
# 0.0.0.0 keeps the image usable under a plain compose network too. Reachability is restricted by
# the compose topology, not by the bind address.
CMD ["sh", "-c", "exec python /srv/private_response_server.py --task-number ${CHEMBENCH_SIDECAR_TASK_NUMBER} --snapshot /snapshot --host 0.0.0.0 --port ${CHEMBENCH_SIDECAR_PORT}"]
