FROM mambaorg/micromamba:2.8.1
USER root
COPY vendor/docker-29.1.3.tgz /tmp/docker-cli.tgz
RUN tar -xzf /tmp/docker-cli.tgz -C /usr/local/bin --strip-components=1 docker/docker && rm /tmp/docker-cli.tgz
COPY vendor/docker-compose-v2.deb /tmp/docker-compose.deb
RUN dpkg-deb -x /tmp/docker-compose.deb /tmp/compose && mkdir -p /usr/local/libexec/docker/cli-plugins /opt/organizer/licenses && cp /tmp/compose/usr/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose && cp /tmp/compose/usr/share/doc/docker-compose-v2/copyright /opt/organizer/licenses/docker-compose-copyright && chmod 755 /usr/local/libexec/docker/cli-plugins/docker-compose
RUN micromamba create -y -n organizer -c conda-forge python=3.14.4 pip && micromamba clean --all --yes
ENV PATH=/opt/conda/envs/organizer/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONDONTWRITEBYTECODE=1
COPY requirements.lock /opt/organizer/requirements.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: --force-reinstall pip==26.2.1 && python -m pip install --no-cache-dir --only-binary=:all: -r /opt/organizer/requirements.lock && python -m pip check
RUN micromamba list -n organizer --explicit > /opt/organizer/conda-explicit.lock
RUN python --version && harbor --version && docker --version && docker compose version
ENTRYPOINT []
CMD ["python"]
