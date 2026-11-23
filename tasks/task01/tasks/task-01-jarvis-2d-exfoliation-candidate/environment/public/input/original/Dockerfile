FROM condaforge/miniforge3:24.11.3-0
WORKDIR /benchmark
COPY environment-linux.yml /tmp/environment-linux.yml
RUN mamba env create -f /tmp/environment-linux.yml && mamba clean -afy
COPY . /benchmark
ENV PATH=/opt/conda/envs/jarvis_2d_exfoliation_benchmark/bin:$PATH
ENV PYTHONUNBUFFERED=1
CMD ["python", "src/main.py", "--config", "configs/default.yaml"]
