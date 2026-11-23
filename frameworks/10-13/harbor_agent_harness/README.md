# MatBench Harbor Tasks 10-13

Task-scoped author delivery aligned to the 2026-09-05 unified Harbor framework.
The collection contains the common Harbor Agent and framework, four runnable task
directories, separate verifier images, organizer-only private evidence replay,
and author validation tools.

Run static validation:

```bash
./chembench validate
python scripts/validate_handoff_10_13.py
```

Run the scientific replay suite in the pinned RDKit/PyBaMM verifier environment:

```bash
python scripts/validate_tasks_10_13_protocol.py
```

Run one task with Harbor:

```bash
harbor run -p tasks/task-10-matbench-bandgap-pipeline \
  -a agents.chembench_agent:ChemBenchAgent
```

The participant environment copies only `environment/public/` into `/workspace`.
Private data, private graders and Oracle fixtures are retained under each verifier
`tests/task_snapshot/` and must never be mounted into the model workspace. Read
`AUTHOR_HANDOFF_PACKAGE.md` and each task's `MIGRATION.md` before changing release
gates or task-owned scientific logic.
