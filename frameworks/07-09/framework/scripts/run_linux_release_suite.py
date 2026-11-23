"""Execute all six real Harbor trials inside the Linux organizer runner."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    rows = []
    for task in (8, 7, 9):
        for wrong in (False, True):
            name = 'release_v2_wrong' if wrong else 'release_v2_oracle'
            command = [sys.executable, str(ROOT/'scripts/run_release_oracle.py'),
                       '--task', str(task), '--name', name,
                       '--sidecar-host', f'matbench-recovered-task{task:02d}', '--port', '8717']
            if wrong:
                command.append('--wrong')
            result = subprocess.run(command, cwd=ROOT)
            rows.append({'task':task, 'job':name, 'exit_code':result.returncode})
            (ROOT/'release_evidence/linux_suite_progress.json').write_text(
                json.dumps(rows, indent=2)+'\n', encoding='utf-8')
            if result.returncode:
                return result.returncode
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
