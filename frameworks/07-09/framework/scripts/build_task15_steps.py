#!/usr/bin/env python3
from pathlib import Path
import json
import sys


TEST = '''#!/bin/bash
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
p=Path("/workspace/rounds/round-{round}/candidate.json")
value=json.loads(p.read_text()) if p.is_file() else None
candidate=value.get("candidate", value) if isinstance(value, dict) else None
if not isinstance(candidate, dict) or not str(candidate.get("smiles", "")).strip():
    raise SystemExit("missing or invalid candidate SMILES")
PY
echo 1 > /logs/verifier/reward.txt
'''


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    task = root / "tasks/task-15-lipophilic-sulfonamide-edit-budget"
    corpus = task / "tests/task_snapshot/private_grader/replay_corpus/deepseek-v4-pro_q1_lipophilic_sulfonamide_edit_budget_20260801_104757"
    replay = {
        int(record["round"]): record["submitted_candidate"]
        for line in (corpus / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and (record := json.loads(line)).get("submitted_candidate")
    }
    for number in range(1, 6):
        step = task / "steps" / f"round-{number}"
        (step / "tests").mkdir(parents=True, exist_ok=True)
        previous = number - 1
        if number == 1:
            intro = (
                "Read `/workspace/input/tasks/q1_lipophilic/question.md` and its public task config.\n"
                "Submit one candidate for round 1.\n"
            )
        else:
            intro = (
                f"Computed public feedback for round {previous} is in "
                f"`/workspace/rounds/round-{previous}/feedback.json`.\n"
                f"Use it to submit one candidate for round {number}.\n"
            )
            workdir = step / "workdir"
            workdir.mkdir(exist_ok=True)
            setup = f'''#!/bin/bash
set -euo pipefail
python /workspace/chembench_protocol/evaluate_round.py \\
  --round {number} \\
  --candidate /workspace/rounds/round-{previous}/candidate.json \\
  --output /workspace/rounds/round-{previous}/feedback.json \\
  --state /workspace/rounds/public_protocol_state.json \\
  --task-config /workspace/input/tasks/q1_lipophilic/task_config.yaml
'''
            (workdir / "setup.sh").write_text(setup, encoding="utf-8", newline="\n")
        instruction = intro + f'''Write `/workspace/rounds/round-{number}/candidate.json` with exactly this public schema:

```json
{{"candidate": {{"name": "...", "smiles": "...", "rationale": "..."}}}}
```

Do not report computed properties as authoritative; the harness recomputes them.
'''
        (step / "instruction.md").write_text(instruction, encoding="utf-8", newline="\n")
        (step / "tests/test.sh").write_text(TEST.format(round=number), encoding="utf-8", newline="\n")
        solution = step / "solution"
        solution.mkdir(exist_ok=True)
        (solution / "candidate.json").write_text(
            json.dumps({"candidate": replay[number]}, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        (solution / "solve.sh").write_text(
            f"#!/bin/bash\nset -euo pipefail\nmkdir -p /workspace/rounds/round-{number}\n"
            f"cp /solution/candidate.json /workspace/rounds/round-{number}/candidate.json\n",
            encoding="utf-8", newline="\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
