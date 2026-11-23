# Release validation record

Release: `1.3`

Checks performed in the build container:

- Python compilation for `src/`, `scripts/`, and `tests/`: passed.
- Release-container pytest: 24 passed, 1 skipped.
- JSON and YAML parsing: passed.
- Public dataset checksum verification: passed.
- Property score weights: 50 points.
- Candidate-library score weights: 50 points, including 10 points for five
  independently re-relaxed CHGNet candidates.
- Secret-like credential scan: no findings in this release.
- Old task/version and local absolute-path residue scan: no findings.

The skipped test requires evaluator-only private labels and is expected in the
public release. The pinned image includes pymatgen, CHGNet, and the OpenAI
client. A five-structure CHGNet relaxation smoke test completed successfully.
Run the complete public suite before leaderboard deployment:

```bash
conda env create -f environment-linux.yml
conda activate jarvis_2d_exfoliation_benchmark
pytest -q
```

Release container tag: `msebench-jarvis-candidate-v0.5.0:latest`.

Re-run the commands above and `python scripts/validate_package.py` after every
release change; this record is not a substitute for current CI output.
