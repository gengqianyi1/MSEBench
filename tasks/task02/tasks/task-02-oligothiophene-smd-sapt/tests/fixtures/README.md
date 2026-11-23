# Task 02 scientific-error fixtures

The fixtures are generated from the organizer-only Oracle workspace at test
time; no private reference files are copied into the public environment. Run
inside `harbor-migration/task02:verifier` with the Oracle workspace mounted
read-only:

```bash
docker run --rm --network none \
  -v "$ORACLE_WORKSPACE:/oracle:ro" \
  -v "$EVIDENCE_DIR:/evidence" \
  harbor-migration/task02:verifier \
  micromamba run -n chembench python /tests/check_scientific_error_fixtures.py \
  --oracle-workspace /oracle --output /evidence/task02_scientific_error_fixtures.json
```

The expected deterministic scores are frozen in the checker:

| fixture | mutation | expected score |
|---|---|---:|
| oracle | no mutation | 98.75 |
| unit_mixup | XYZ coordinates interpreted as bohr instead of Å | 95.817 |
| solvent_order | n-hexane/chloroform converged energies swapped | 90.75 |
| sapt_dominance | n-hexane attractive component falsely reported as electrostatics | 25.0 |

All runs use `--network none` and the existing verifier image; no additional
image or package is required.
