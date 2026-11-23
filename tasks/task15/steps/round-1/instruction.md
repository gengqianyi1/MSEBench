Read `/workspace/input/tasks/q1_lipophilic/question.md` and its public task config.
Submit one candidate for round 1.
Write `/workspace/rounds/round-1/candidate.json` with exactly this public schema:

```json
{"candidate": {"name": "...", "smiles": "...", "rationale": "..."}}
```

Do not report computed properties as authoritative; the harness recomputes them.
