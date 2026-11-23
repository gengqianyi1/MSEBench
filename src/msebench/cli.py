"""Run MSEBench TextQA and local material-task evaluations."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import textqa


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["textqa"]:
        return textqa.main(args[1:])

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run 'msebench textqa --help' for offline or model evaluation.",
    )
    parser.add_argument("--workspace", type=Path, help="Assembled MSEBench workspace containing run.py")
    options, runner_args = parser.parse_known_args(args)
    if not runner_args:
        parser.print_help()
        return 0
    configured = options.workspace or os.getenv("MSEBENCH_INTEG_ROOT")
    workspace = Path(configured).resolve() if configured else Path(__file__).resolve().parents[2]
    runner = workspace / "run.py"
    if not runner.is_file() or not (workspace / "frameworks").is_dir():
        parser.error("material tasks require --workspace pointing to the assembled MSEBench workspace")
    return subprocess.call([sys.executable, str(runner), *runner_args])


if __name__ == "__main__":
    raise SystemExit(main())
