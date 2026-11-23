"""Resolve integrated tasks without author-local symlinks or duplicated sources."""
from pathlib import Path

FRAMEWORK = Path(__file__).resolve().parents[1]
PACKAGE = FRAMEWORK.parents[1]
SLUGS = {19: 'task-19-active-impurity-elucidation', 20: 'task-20-catalyst-deactivation-diagnosis'}

def task_path(task):
    number = task if isinstance(task, int) else next(n for n, slug in SLUGS.items() if slug == task)
    path = PACKAGE / 'tasks' / f'task{number}'
    if not path.is_dir():
        raise FileNotFoundError(f'Missing integrated task: {path}')
    return path
