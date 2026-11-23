"""Read-only standalone package/dependency preflight; no provider calls."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    task_id = (ROOT / "TASK_ID").read_text().strip()
    tasks = list((ROOT / "tasks").glob("task-*"))
    if len(tasks) != 1 or not tasks[0].name.startswith(f"task-{task_id}-"):
        raise RuntimeError("Package must contain exactly its own task")
    task = tasks[0]
    if importlib.metadata.version("harbor") != "0.21.0":
        raise RuntimeError("Harbor version mismatch")
    for relative in ("agents/chembench_agent.py", "agents/workspace_sidecar_agent.py",
                     "verifiers/workspace_sidecar_verifier.py", "scripts/run_workspace_harbor.py",
                     "scripts/participant_fixture.py", "templates/sidecar/workspace_sidecar_server.py"):
        if not (ROOT / relative).is_file():
            raise RuntimeError(f"Missing framework file: {relative}")
    for relative in ("tests/protocol_grader.py", "tests/private_suite_adapter.py",
                     "tests/test.sh", "environment/environment.lock.yml"):
        if not (task / relative).is_file():
            raise RuntimeError(f"Missing task file: {relative}")
    cfg = json.loads((task / "environment/public/.chembench_agent.json").read_text())
    for schema in (task / "environment/public").rglob("*.schema.json"):
        import jsonschema
        document = json.loads(schema.read_text())
        jsonschema.validators.validator_for(document).check_schema(document)
    task_toml = tomllib.loads((task / "task.toml").read_text())
    for name, image_id in json.loads((ROOT / "docker/image_ids.json").read_text()).items():
        actual = subprocess.check_output(["docker", "image", "inspect", name, "--format", "{{.Id}}"], text=True).strip()
        if actual != image_id:
            raise RuntimeError(f"Image missing/changed: {name}; run bash run.sh load")
    for name in ("agents.chembench_agent", "verifiers.workspace_sidecar_verifier"):
        module = __import__(name, fromlist=["*"])
        if ROOT not in Path(module.__file__).resolve().parents:
            raise RuntimeError(f"Framework import escaped this package: {name}")
    local_dependencies = []
    for dist in importlib.metadata.distributions():
        direct = dist.read_text("direct_url.json")
        if direct:
            value = json.loads(direct)
            if value.get("dir_info", {}).get("editable") or value.get("url", "").startswith("file:"):
                local_dependencies.append(dist.metadata["Name"])
    if local_dependencies:
        raise RuntimeError(f"Local Python dependencies found: {local_dependencies}")
    print(json.dumps({"valid": True, "task": task.name, "harbor": "0.21.0",
                      "python": sys.version.split()[0], "task_version": task_toml["task"]["version"],
                      "required_artifacts": cfg["required_artifacts"], "local_python_dependencies": []}))


if __name__ == "__main__":
    main()
