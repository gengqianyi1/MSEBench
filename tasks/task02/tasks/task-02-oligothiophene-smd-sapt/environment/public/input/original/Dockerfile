FROM condaforge/miniforge3:24.11.3-0

WORKDIR /benchmark

COPY environment-linux.yml /tmp/environment-linux.yml
COPY scripts/check_environment.py /tmp/check_environment.py

SHELL ["/bin/bash", "-lc"]

RUN mamba env create -f /tmp/environment-linux.yml && mamba clean -afy

ENV PATH=/opt/conda/envs/chembench_task2_tool/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# The conda-forge PySCF package can import successfully but may be built
# without libsolvent/SMD support. Build PySCF from source with SMD enabled.
ARG PYSCF_REF=v2.13.1
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate chembench_task2_tool && \
    python -m pip uninstall -y pyscf pyscf-forge || true && \
    CMAKE_CONFIGURE_ARGS="-DENABLE_SMD=ON -DBUILD_MARCH_NATIVE=OFF -DENABLE_XCFUN=OFF -DBUILD_XCFUN=OFF" \
      python -m pip install --no-cache-dir --no-binary pyscf \
      "git+https://github.com/pyscf/pyscf.git@${PYSCF_REF}" && \
    python /tmp/check_environment.py --quick

COPY . /benchmark

CMD ["python", "src/main.py", "--config", "configs/default.yaml"]
