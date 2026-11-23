FROM mambaorg/micromamba:2.8.1
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.lock.yml /tmp/environment.lock.yml
RUN micromamba create -y -n chembench -f /tmp/environment.lock.yml && micromamba clean --all --yes
ENV PATH=/opt/conda/envs/chembench/bin:$PATH
ENV MAMBA_DEFAULT_ENV=chembench
USER root
RUN mkdir -p /tests /task /workspace /logs/verifier /run && chown -R $MAMBA_USER:$MAMBA_USER /tests /task /workspace /logs /run
USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER task_snapshot/ /task/
COPY --chown=$MAMBA_USER:$MAMBA_USER run_legacy_grader.py /tests/run_legacy_grader.py
COPY --chown=$MAMBA_USER:$MAMBA_USER private_suite_adapter.py /tests/private_suite_adapter.py
COPY --chown=$MAMBA_USER:$MAMBA_USER protocol_grader.py /tests/protocol_grader.py
COPY --chown=$MAMBA_USER:$MAMBA_USER test.sh /tests/test.sh
RUN chmod +x /tests/test.sh
WORKDIR /workspace
CMD ["bash"]
