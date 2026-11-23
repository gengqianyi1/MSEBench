# Coupled descriptor-window molecular optimization with compensating edits — public side

An iterative agent benchmark that measures whether an LLM can raise a molecule's **QED**
(a computed drug-likeness *surrogate*) while **simultaneously** satisfying several competing hard
constraints: three narrow property windows, a synthetic-accessibility cap, a zero-structural-alert
requirement, and a Morgan-fingerprint similarity floor that acts as an edit budget.

The windows are deliberately **narrow and coupled**, and the starting molecule sits close to an edge
of each of them, so the edits that most easily raise QED push molecular weight, LogP and polar
surface area out of range at the same time. A QED gain has to be paid for with a compensating
structural change. The model reasons from returned numbers alone — no strategy hints, no
pass/fail flags.

**This half of the package contains no answer.** There is no reference target, no reference
structure, no score band table and no scorer here, and this harness does not compute a score. See
`../grader/README.md` for the grading side.

## Positioning — what this measures and what it does not

Every number is computed deterministically by RDKit from a SMILES string. QED is a **surrogate**
property, not a measurement: it is a published desirability function over eight descriptors. The
benchmark measures constrained multi-site molecular editing under numeric feedback. It does **not**
measure potency, selectivity, ADMET, developability or real synthetic feasibility, and it contains
no experimental or simulated assay data of any kind.

## Task Summary

- 5 rounds, exactly one submitted SMILES per round.
- After each round, RDKit evaluates admissibility, MW, LogP, TPSA, QED, SA score, structural alerts
  (PAINS + Brenk + NIH), and Morgan-fingerprint similarity to the starting molecule.
- The evaluator returns **computed property values only** — no per-constraint pass/fail flags and
  no success verdict.
- A submission that is not exactly one neutral organic molecule (salt, mixture, ion, isotope label,
  atom map, wildcard, radical, non-organic element) is **rejected** with a machine-readable reason
  code and cannot score.

## File Structure

```text
configs/default.yaml                   run config (also default_deepseek / default_kimi / default_glm)
configs/api_config_*.example.json      endpoint templates, one per official model; copy and fill in
                                       your own key. default_qwen.yaml and its template are RETIRED
                                       (first round only) and must not enter a comparison table.
tasks/q1_multiobjective/question.md    the exact text shown to the model
tasks/q1_multiobjective/task_config.yaml   windows, iteration budget, feedback policy
src/main.py                            the run harness — runs rounds, does NOT score
src/llm_client.py                      OpenAI-compatible client with retry/truncation handling
src/evaluator.py                       RDKit property evaluation and constraint flags
src/validation.py                      input admissibility gate and reason codes
src/objective.py                       objective selection helpers (no reference, no bands)
src/feedback.py                        reduced-bandwidth round feedback
src/parser.py                          candidate JSON parsing
src/markdown_report.py, src/utils.py
scripts/test_llm_connection.py         endpoint smoke test
scripts/trajectory_to_md.py            readable trajectory export
scripts/trajectory_to_txt.py
tests/test_input_validation.py         admissibility + constraint-boundary regression suite
tests/test_harness_robustness.py       parser / empty response / duplicate / abnormal-input suite
requirements.txt
```

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate    # any CPython >= 3.10
pip install -r requirements.txt
# SA_Score ships inside the RDKit Contrib directory; no extra install is needed.

# The official second-round model set is deepseek-v4-pro / kimi-k2.6 / GLM-5.2, all at
# temperature 0.0. Only the deepseek endpoint is verified -- read `_endpoint_status` in the kimi
# and glm templates before spending a run on either.
cp configs/api_config_deepseek.example.json configs/api_config_deepseek.json
cp configs/api_config_kimi.example.json     configs/api_config_kimi.json
cp configs/api_config_glm.example.json      configs/api_config_glm.json
# edit each copy and replace PUT_YOUR_..._KEY_HERE with the real key
```

## Run

```bash
python src/main.py --config configs/default_deepseek.yaml
python src/main.py --config configs/default_kimi.yaml
python src/main.py --config configs/default_glm.yaml
```

Each run creates a timestamped folder under `runs/` containing `trajectory.jsonl`,
`llm_api_debug.jsonl`, `conversation.md` and `run_summary.json`. The copied
`api_config_used.json` has its API key redacted. **`run_summary.json` carries no score** — grading
is a separate, offline step performed by `../grader/scripts/score_run.py`.

## Checks you can run without an API key

```bash
python tests/test_input_validation.py     # admissibility + constraint-window boundaries
python tests/test_harness_robustness.py   # abnormal input costs a round, never the run
```

`test_harness_robustness.py` drives a complete five-round run against a scripted LLM stub in which
every round is pathological (unparseable text, empty SMILES, unparseable SMILES, an LLM failure, a
salt) and asserts that all five rounds still complete and are recorded. That is the direct answer to
"can abnormal values crash the benchmark or end it early?" — they cannot.

## Scoring

Total score: 100 points, and **all 100 come from the objective**.

### Constraint satisfaction is a gate, not a scoring category: 0 points

A molecule that violates any constraint is not a usable design and contributes nothing. A molecule
that satisfies all of them earns **no bonus for doing so** — staying inside the windows is the entry
ticket, not the achievement being measured. The per-constraint results are still computed and
reported as diagnostics.

The constraints (all hard, all zero-point gates) are: one admissible neutral organic molecule;
molecular weight **280–320** (narrow); LogP **2.0–3.0** (narrow); TPSA **30–100** (narrow);
SA score ≤ 3.5; zero structural alerts (PAINS/Brenk/NIH); Morgan similarity to the starting
molecule ≥ 0.44.

### Optimality: 100 points

The best **usable** molecule of the trajectory — feasible **and** with QED above the starting
molecule's — is compared against a reference target held by the grading side:

```text
achievement = (best_usable_QED - start_QED) / (reference_QED - start_QED)
```

Points are a monotone graded function of `achievement`. **The reference value and the band table are
not in this half of the package**; they live in `grader/reference/reference_target.yaml`. Achievement
is clipped at 1.0, so a model that meets or beats the reference earns full marks rather than being
penalised.

## Environment, tools and licensing

Install with `pip install -r requirements.txt`. Every dependency is **pinned to the exact version the
shipped trajectories and gate measurements were produced with**. No compiled binaries, system
packages, or proprietary software are needed — everything is pure pip and permissively licensed.

| Component | Version pinned | License | Used for |
|---|---|---|---|
| Python | 3.10.19 | PSF | runtime |
| RDKit | 2025.9.6 | BSD-3-Clause | descriptors, QED, Morgan fingerprints, PAINS/Brenk/NIH FilterCatalog, MolStandardize |
| RDKit Contrib SA_Score | bundled with RDKit | BSD-3-Clause | synthetic accessibility (Ertl & Schuffenhauer, 2009) |
| openai | 2.15.0 | Apache-2.0 | OpenAI-compatible chat client |
| pyyaml | 6.0.3 | MIT | config loading |
| numpy | 2.2.6 | BSD-3-Clause | numerics |

Reference platform: macOS 26.5 arm64. Nothing is platform-specific.

All evaluation is deterministic: RDKit descriptors, QED, SA score, the alert catalogs and the
fingerprint similarity are pure functions of the submitted SMILES, so re-scoring any trajectory
reproduces the same numbers exactly.

## Harness robustness

A benchmark round must fail only because of the model's chemistry, never because of the plumbing.

- **Truncated responses.** A reasoning model can emit a well-formed JSON object and then run out of
  completion budget mid-string. The client detects `finish_reason == "length"` with non-empty
  content, retries with the next larger budget, and only gives up after the schedule is exhausted
  (recorded as `n_truncated_llm_responses`).
- **Timeouts escalating themselves.** Retrying a timeout with a *larger* `max_tokens` makes the next
  request slower. The completion budget now escalates only for failures a bigger budget can fix
  (truncation, reasoning-exhausted empty content); transport errors retry at the same budget.
- **Abnormal candidates.** Every malformed message, invalid SMILES and inadmissible input costs the
  round and nothing more; `tests/test_harness_robustness.py` pins that.

`run_summary.json → checks` exposes `n_parse_failures`, `n_llm_call_failures`,
`n_empty_llm_responses`, `n_truncated_llm_responses` and `n_rejected_by_input_validation` so a
reviewer can confirm a trajectory was not degraded by infrastructure.
