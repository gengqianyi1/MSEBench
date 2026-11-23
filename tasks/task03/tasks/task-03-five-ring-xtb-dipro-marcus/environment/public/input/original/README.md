# Five-Ring xTB-DIPRO Marcus Hole-Rate Benchmark

Benchmark release: **1.3**

This benchmark is a fixed-round closed-loop molecular design task for small-molecule organic semiconductor backbones.

The LLM only proposes molecular modifications. The evaluator handles:

- RDKit validation;
- five-ring fused-core detection;
- multi-conformer 3D generation and force-field conformer selection;
- xTB neutral/cation geometry optimization;
- four-point hole reorganization energy;
- lowest-energy slipped cofacial dimer packing search;
- xTB DIPRO hole transfer integral;
- Marcus hole hopping rate;
- scoring and logging.

## Folder structure

```text
five_ring_xtb_dipro_marcus_benchmark/
  configs/
    default.yaml
    api_config.json
    api_config.example.json
  tasks/
    five_ring_xtb_dipro_marcus/
      question.md
      task_config.yaml
  src/
    main.py
    llm_client.py
    parser.py
    evaluator.py
    scoring.py
    feedback.py
    xtb_runner.py
    geometry.py
    chemistry.py
    markdown_report.py
    utils.py
  scripts/
    validate_smiles.py
    trajectory_to_md.py
  runs/
  requirements.txt
```

## Installation

Recommended Python version: 3.10.

Create the tested conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate five-ring-bench
python -c "import numpy, openai, yaml, rdkit; print('tools ok')"
xtb --version
```

The pinned tested versions are:

```text
python=3.10.20
numpy=2.2.6
openai==2.43.0
pyyaml=6.0.3
rdkit=2026.03.1
xtb=6.7.1=h8876d29_4
```

This benchmark calls xTB as an external executable. If xTB is missing, RDKit validation will still work, but xTB/DIPRO candidates will be marked as failed computed candidates.

## Configure LLM API

Edit:

```text
configs/api_config.json
```

Example:

```json
{
  "provider": "openai_compatible",
  "model": "deepseek-v4-pro",
  "api_key": "YOUR_API_KEY",
  "base_url": "https://api.example.com/v1",
  "temperature": 0.7,
  "max_tokens": 4096,
  "max_tokens_schedule": [4096, 8192, 12000],
  "max_retries": 2,
  "retry_delay_seconds": 2,
  "timeout": 60,
  "save_full_api_response": true
}
```

## End-to-end evaluation

After configuring the API, run one model-specific experiment:

```bash
python src/main.py --config configs/deepseek.yaml
# or: python src/main.py --config configs/qwen.yaml
# or: python src/main.py --config configs/glm.yaml
```

A timestamped run directory will be created under `runs/`.

For an isolated container run:

```bash
docker build -t five-ring-bench:1.3 .
docker run --rm -it five-ring-bench:1.3 \
  python src/main.py --config configs/default.yaml
```

The evaluator computes and scores each proposal during the fixed five rounds;
there is no separate private scoring command. A complete run must contain
`final_report.json` and `trajectory.jsonl`. The authoritative score is:

```text
runs/<run_id>/final_report.json
  score.total_score
```

Also report `score.eligible`, `score.rate_performance_score`,
`score.n_successful_computed_candidates`, `score.n_target_hits`, and the
recommended candidate rate. Constraints are eligibility gates and contribute
no separate points. A process exit without `final_report.json` is a failed
run and must not be reported as zero-scoring completion.

Each run directory contains:

```text
trajectory.jsonl
llm_api_debug.jsonl
conversation.md
final_report.json
question.md
task_config.yaml
config_used.yaml
api_config_used.json
```

## Human-readable report

Open:

```text
runs/<run_id>/conversation.md
```

It contains all LLM messages, raw outputs, parsed candidates, evaluator results, feedback, best-so-far information, and final score.

## Validate one SMILES without LLM

```bash
python scripts/validate_smiles.py --smiles "..."
```

This runs the RDKit/fused-ring constraint checks only by default.

To also run xTB/DIPRO:

```bash
python scripts/validate_smiles.py --smiles "..." --run-xtb
```

## Convert trajectory to Markdown

```bash
python scripts/trajectory_to_md.py runs/<run_id>/trajectory.jsonl
```

## Important notes

- The LLM must not invent computed properties.
- The evaluator is the source of truth for `lambda_h_eV`, `Vh_DIPRO_eV`, and `marcus_hole_rate_s_inv`.
- The five-ring fused-core constraint is algorithmic: rings are connected as fused when they share at least two atoms.
- No rings may occur outside the single five-ring fused cluster.
- Scoring is based on hard constraints and best Marcus hole hopping rate.

## Release validation and citation

```bash
python scripts/check_environment.py
python scripts/build_manifest.py
python scripts/validate_package.py
python -m unittest discover -s tests -v
```

See `REPRODUCIBILITY.md`, `SECURITY_AND_PRIVATE_DATA.md`, `CITATION.cff`, and
`THIRD_PARTY_LICENSES.md` for the release contracts.


## Windows xTB output encoding note

On Chinese Windows, Python may decode subprocess output with GBK by default. xTB output can contain bytes that GBK cannot decode, which may produce errors such as:

```text
UnicodeDecodeError: 'gbk' codec can't decode byte ...
unsupported operand type(s) for +: 'NoneType' and 'str'
```

This patched version captures xTB stdout/stderr as bytes and decodes with `errors="replace"` to avoid that issue.
