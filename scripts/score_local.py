"""Score a saved material-task submission with its locked verifier image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_TASKS = {4, 5, 6, 7, 8, 9, 18}


def score(task: int, submission: Path, output: Path, evidence: Path | None = None) -> int:
    submission, output = submission.resolve(strict=True), output.absolute()
    if not submission.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("Provide a submission directory and a new output directory")
    if output.is_relative_to(submission):
        raise ValueError("Output must be outside the submission directory")
    if task in EVIDENCE_TASKS and evidence is None:
        raise ValueError("This task requires --evidence from the original local evaluation run")
    if any(path.is_symlink() for path in submission.rglob("*")):
        raise ValueError("Submission must contain regular files without symbolic links")
    if evidence is not None:
        evidence = evidence.resolve(strict=True)
        if not evidence.is_dir() or output.is_relative_to(evidence):
            raise ValueError("Evidence must be a directory outside the output")
        if any(path.is_symlink() for path in evidence.rglob("*")):
            raise ValueError("Evidence must contain regular files without symbolic links")
    lock = json.loads((ROOT / f"images/task{task:02}/images.lock.json").read_text())
    image = lock["images"].get(f"task{task:02}-verifier.tar")
    if image is None and task in {7, 8, 9}:
        image = lock["images"].get(f"task{task:02}-sidecar.tar")
    if image is None:
        raise ValueError("Task has no locked verifier image")
    inspect = subprocess.run(
        ["docker", "image", "inspect", image["tag"], "--format", "{{.Id}}"],
        capture_output=True, text=True, check=True,
    )
    if inspect.stdout.strip() != image["image_id"]:
        raise ValueError("Loaded verifier image does not match the image lock")
    output.mkdir(parents=True)
    shutil.copytree(submission, output / "submission")
    (output / "logs").mkdir()
    (output / "run").mkdir()
    command = ["docker", "run", "--rm", "--network", "none", "--entrypoint", "bash"]
    for source, target in ((output / "submission", "/workspace"), (output / "logs", "/logs/verifier"),
                           (output / "run", "/run")):
        command += ["--mount", f"type=bind,source={source},target={target}"]
    if evidence:
        shutil.copytree(evidence, output / "evidence")
        command += ["--mount", f"type=bind,source={output / 'evidence'},target=/organizer_evidence"]
        if 10 <= task <= 13:
            command += ["--env", "CHEMBENCH_REPLAY_EVIDENCE=/organizer_evidence"]
    command += [image["tag"], "/tests/test.sh"]
    result = subprocess.run(command, capture_output=True, text=True)
    (output / "stdout.txt").write_text(result.stdout)
    (output / "stderr.txt").write_text(result.stderr)
    reward = output / "logs/reward.txt"
    record = {
        "task": task, "verifier_image_id": image["image_id"], "exit_code": result.returncode,
        "reward": reward.read_text().strip() if reward.is_file() else None,
        "evidence_supplied": evidence is not None,
    }
    (output / "score_run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=range(1, 21), required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, help="Saved runtime evidence for this submission")
    args = parser.parse_args()
    try:
        raise SystemExit(score(args.task, args.submission, args.output, args.evidence))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Local scoring failed: {exc}\n")


if __name__ == "__main__":
    main()
