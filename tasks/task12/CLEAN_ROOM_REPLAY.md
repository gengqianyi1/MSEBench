# Task 12 clean-room replay and environment fingerprint

This record implements the 2026-08-26 recheck request to make the lockfile the clean-room reference and retain an environment hash.

## Authoritative locks

- Agent lock: `environment/environment.lock.yml`
- Verifier lock: `tests/environment.lock.yml`
- SHA-256 of each lock: `03704a4b9218ccce46b23f54ef947bdd140233511d4054f3d4a4166f6ef0f4c8`
- Expected runtime: Python 3.9.24, PyBaMM 25.4.2, NumPy 1.26.4, SciPy 1.13.1, Casadi 3.6.7.
- Scientific contract: SPMe, Chen2020, Casadi safe mode, rtol/atol 1e-7, one continuous CC/CV/rest/discharge experiment.

## Clean-room commands

From this task directory:

```bash
docker build --no-cache -t chembench-task12-verifier:revision4 -f tests/Dockerfile tests
docker image inspect chembench-task12-verifier:revision4 --format '{{.Id}}'
harbor run -p . -a ../agents/chembench_agent.py --model <same-model-id> --env CHEMBENCH_TEMPERATURE=0
```

Save the Docker image ID, Harbor version, model/provider ID, decoding parameters, run IDs and both legacy/Harbor scores beside `tests/environment_fingerprint_20260831.json`.

## 2026-08-31 offline replay

The local `openchemie` environment matched every pinned scientific package version and ran the Oracle plus three invalid cases. Oracle raw score was 80.0 and legacy/evidence-replay delta was 0.0. The host was Windows, so this is a scientific-runtime replay, not the requested Linux Docker clean room. Docker, Harbor and model credentials were unavailable; image-digest and live trajectory-parity fields therefore remain pending and release gates were not changed.

