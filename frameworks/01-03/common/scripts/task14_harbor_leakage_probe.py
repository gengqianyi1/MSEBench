#!/usr/bin/env python3
"""Create/run a disposable Task 14 copy instrumented for shared-verifier residue."""
from __future__ import annotations
import argparse, json, pathlib, shutil, subprocess, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tasks" / "task-14-multiobjective-mol-design"
PROBE_TEMPLATE = r"""
# --- ChemBench development-only shared-verifier residue probe ---
python - <<'PYPROBE'
import hashlib, json
from pathlib import Path
round_name = __ROUND_NAME__
out = Path('/workspace/harbor_runtime_probe')
out.mkdir(parents=True, exist_ok=True)
paths = [Path('/tests/test.sh'), Path('/logs/verifier/reward.txt'), Path('/logs/verifier/reward.json')]
rec = {'before_agent_step': round_name, 'paths': {}}
for p in paths:
    item = {'exists': p.exists(), 'is_file': p.is_file()}
    if p.is_file():
        b = p.read_bytes()
        item.update({'size': len(b), 'sha256': hashlib.sha256(b).hexdigest()})
    rec['paths'][str(p)] = item
(out / f'{round_name}.json').write_text(json.dumps(rec, indent=2), encoding='utf-8')
PYPROBE
# --- end probe ---
"""

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=pathlib.Path)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--keep", action="store_true")
    args, extra = ap.parse_known_args()
    auto = args.output is None
    parent = pathlib.Path(tempfile.mkdtemp(prefix="chembench-task14-probe-")) if auto else args.output.parent
    out = (parent / SOURCE.name) if auto else args.output
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(SOURCE, out)
    # Ensure the development-only probe output is collected as a Harbor artifact.
    task_toml = out / "task.toml"
    toml_text = task_toml.read_text(encoding="utf-8")
    if '"/workspace/harbor_runtime_probe"' not in toml_text:
        toml_text = toml_text.replace(
            'artifacts = [\n',
            'artifacts = [\n    "/workspace/harbor_runtime_probe",\n',
            1,
        )
        task_toml.write_text(toml_text, encoding="utf-8", newline="\n")

    for n in range(2, 6):
        setup = out / "steps" / f"round-{n}" / "workdir" / "setup.sh"
        text = setup.read_text(encoding="utf-8")
        marker = PROBE_TEMPLATE.replace("__ROUND_NAME__", repr(f"round-{n}"))
        lines = text.splitlines(True)
        idx = 1 if lines and lines[0].startswith("#!") else 0
        if idx < len(lines) and "set " in lines[idx]: idx += 1
        setup.write_text("".join(lines[:idx]) + marker + "".join(lines[idx:]), encoding="utf-8", newline="\n")
    print(json.dumps({
        "source_task": str(SOURCE),
        "instrumented_task": str(out),
        "probe_files_expected": [f"/workspace/harbor_runtime_probe/round-{n}.json" for n in range(2, 6)],
        "note": "Canonical task is untouched; use Oracle so the probe cannot affect an evaluated model."
    }, indent=2))
    rc = 0
    if args.run:
        if shutil.which("harbor") is None: raise SystemExit("harbor CLI not found")
        cmd = ["harbor", "run", "-p", str(out), "-a", "oracle", *extra]
        print("+", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd)
    if auto and not args.keep and not args.run: shutil.rmtree(parent, ignore_errors=True)
    elif auto: print(f"Probe copy retained at: {out}")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
