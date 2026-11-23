Read `/workspace/input/tasks/q1_multiobjective/question.md` (all constraint windows are stated there).
The starting SMILES and its RDKit properties are in `/workspace/input/tasks/q1_multiobjective/task_config.yaml`.
For the round-completeness and reward rules, see the iteration section of that task configuration:
all five rounds are mandatory, and a run with no parseable candidate in any round is INCOMPLETE
and receives reward 0.

Submit exactly ONE candidate for round 1 by writing the file
`/workspace/rounds/round-1/candidate.json` with this schema:

```json
{"candidate": {"name": "...", "smiles": "...", "rationale": "..."}}
```

One molecule only. Do not include pass/fail text.