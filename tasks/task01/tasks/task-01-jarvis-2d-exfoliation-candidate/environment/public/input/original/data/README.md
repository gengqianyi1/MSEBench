# Public data files

- `train.json.gz`: labeled training records, gzip JSON Lines.
- `validation.json.gz`: labeled validation records, gzip JSON Lines.
- `test_public.json.gz`: unlabeled public test records, gzip JSON Lines.
- `dataset_metadata.json`: frozen data and property-scoring metadata.
- `candidate_generation_constraints.json`: public hard limits and deterministic candidate-scoring thresholds.
- `checksums.sha256`: public file hashes.

The agent workspace contains only these public files. Private property labels are supplied later to the isolated scorer. Candidate-library quality is evaluated from structures and generator artifacts; generated-candidate exfoliation energy is not scored.
