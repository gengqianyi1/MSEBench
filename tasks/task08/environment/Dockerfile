FROM mambaorg/micromamba:2.8.1
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.lock.yml /tmp/environment.lock.yml
RUN micromamba create -y -n chembench -f /tmp/environment.lock.yml && micromamba clean --all --yes
ENV PATH=/opt/conda/envs/chembench/bin:$PATH
ENV MAMBA_DEFAULT_ENV=chembench
USER root
COPY conda /usr/local/bin/conda
RUN chmod 755 /usr/local/bin/conda
RUN mkdir -p /workspace /logs/agent /logs/artifacts /logs/verifier && chown -R $MAMBA_USER:$MAMBA_USER /workspace /logs
USER $MAMBA_USER
WORKDIR /workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER public/ /workspace/
CMD ["bash"]
