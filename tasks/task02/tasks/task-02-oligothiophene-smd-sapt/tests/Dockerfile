FROM mambaorg/micromamba:2.8.1
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.lock.yml /tmp/environment.lock.yml
RUN micromamba create -y -n chembench -f /tmp/environment.lock.yml && micromamba clean --all --yes
COPY wheels/pyscf-2.8.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tmp/pyscf-2.8.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN echo '4BAA59297DE8A2992E8AFA5B20A12164B6128C14CCC09431D05F73E3EEFDF008  /tmp/pyscf-2.8.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl' | sha256sum -c - && \
    micromamba run -n chembench python -m pip install --no-cache-dir /tmp/pyscf-2.8.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl && \
    micromamba clean --all --yes
ENV PATH=/opt/conda/envs/chembench/bin:$PATH
ENV MAMBA_DEFAULT_ENV=chembench
USER root
RUN mkdir -p /tests /task /workspace /logs/verifier /run && chown -R $MAMBA_USER:$MAMBA_USER /tests /task /workspace /logs /run
USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER task_snapshot/ /task/
COPY --chown=$MAMBA_USER:$MAMBA_USER run_legacy_grader.py /tests/run_legacy_grader.py
COPY --chown=$MAMBA_USER:$MAMBA_USER private_suite_adapter.py /tests/private_suite_adapter.py
COPY --chown=$MAMBA_USER:$MAMBA_USER sidecar.py /tests/sidecar.py
COPY --chown=$MAMBA_USER:$MAMBA_USER check_scientific_error_fixtures.py /tests/check_scientific_error_fixtures.py
COPY --chown=$MAMBA_USER:$MAMBA_USER test.sh /tests/test.sh
RUN chmod +x /tests/test.sh
WORKDIR /workspace
RUN micromamba run -n chembench python -c "import pyscf; from pyscf.solvent import smd; print('PySCF', pyscf.__version__, 'SMD_OK')"
CMD ["bash"]
