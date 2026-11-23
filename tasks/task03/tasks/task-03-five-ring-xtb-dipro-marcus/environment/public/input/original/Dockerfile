FROM condaforge/miniforge3:24.11.3-0

WORKDIR /benchmark
COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && mamba clean -afy
COPY . /benchmark

ENV PATH=/opt/conda/envs/five-ring-bench/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

CMD ["python", "src/main.py", "--config", "configs/default.yaml"]
