# Task 02 participant environment

The MSEBench Docker image manifest records the tag and SHA256 for
`task02-environment.tar`. After obtaining and verifying the archive, load it on
a compatible Linux Docker host:

```bash
docker load -i task02-environment.tar
```

Use the [Task 02 instruction](../../../../instruction.md) and
[tool policy](../TOOL_POLICY.md) for the staged workflow. Official end-to-end
evaluation uses the organizer's scoring environment.
