FROM python:3.11.9-slim
WORKDIR /tests
COPY environment.lock.yml /tests/environment.lock.yml
COPY private_suite_adapter.py protocol_grader.py /tests/
COPY test.sh /tests/test.sh
RUN chmod +x /tests/test.sh
COPY task_snapshot /task
CMD ["python", "/tests/protocol_grader.py"]
