FROM python:3.11.9-slim
WORKDIR /private
COPY tests/sidecar/hidden_suite_server.py /private/hidden_suite_server.py
COPY tests/private_suite_adapter.py /private/private_suite_adapter.py
COPY tests/task_snapshot /private/task_snapshot
RUN mkdir -p /organizer_evidence
EXPOSE 8820
HEALTHCHECK --interval=5s --timeout=3s --retries=12 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8820/health', timeout=2)"
ENTRYPOINT ["python", "/private/hidden_suite_server.py", "--adapter", "/private/private_suite_adapter.py", "--snapshot", "/private/task_snapshot", "--evidence-dir", "/organizer_evidence", "--host", "0.0.0.0", "--port", "8820"]
