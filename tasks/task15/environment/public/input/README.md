# F (public side) — Surrogate-property optimization of an anilide sulfonamide under a fingerprint-similarity edit budget

**This is the model-visible half of the benchmark.** It contains the task, the run harness, the
property evaluator, the diagnostic tool layer and the regression tests. It contains **no reference
answer, no reference target value, no score bands and no scorer**. Those live in the separate
`grader/` package, and grading is a separate step run there.

## What this benchmark measures, and what it does not

This is a **surrogate-property** benchmark. Every number in it — the objective and all six
constraints — is computed deterministically by RDKit from a SMILES string. Nothing is measured,
assayed, simulated or fitted to project data.

| | |
|---|---|
| **Measured** | constrained multi-site molecular editing: raising a computed drug-likeness surrogate (RDKit QED) under three property windows, a synthetic-accessibility cap, a zero-structural-alert requirement and a Tanimoto similarity floor that acts as an **edit budget**; plus tool selection, parameterisation and error recovery under a per-round quota |
| **Not measured** | biological potency, selectivity, metabolic stability, toxicity, developability, real synthetic feasibility |
| **Data provenance** | 100 % RDKit-computed from SMILES. No experimental, vendor, registry or simulated assay data, and no fitted SAR model, is used anywhere. There is no noise model because there is no measured quantity. |

QED is a published desirability composite fitted to marketed-drug property distributions — a coarse
*surrogate* for drug-likeness, not a measure of candidate quality. Morgan/Tanimoto similarity is a
substructure-overlap statistic used here as an edit budget, not a claim about pharmacological or
synthetic relatedness. SA_Score is a fragment-frequency heuristic, not a route assessment. A score on
this task therefore supports claims about **constrained molecular editing under quantitative
feedback**, and not about medicinal-chemistry judgement or real lead optimization.

## Layout

```
tasks/q1_lipophilic/
    question.md              the prompt shown to the model (self-contained; states every constraint)
    task_config.yaml         start molecule, admissibility rules, constraint windows, tool budget
src/
    evaluator.py             RDKit property computation + per-constraint flags
    validation.py            candidate admissibility: one neutral organic molecule, then standardised
    tools.py                 the four diagnostic tools, the per-round quota and the failure returns
    feedback.py              reduced-bandwidth feedback: computed values only
    parser.py                strict JSON extraction: a turn is tool calls or a submission
    objective.py             objective-field helpers (no reference target involved)
    llm_client.py            OpenAI-compatible client with truncation/timeout retry handling
    main.py                  the run harness -- runs rounds and logs them; does NOT score
    markdown_report.py       human-readable conversation transcript
    utils.py                 config/IO helpers
scripts/
    trajectory_to_md.py      render a trajectory as Markdown
    trajectory_to_txt.py     render a trajectory as plain text
    test_llm_connection.py   one-shot endpoint check
tests/
    test_input_validation.py 123 assertions: salts, mixtures, ions, valences, isotopes, boundaries
    test_tools.py            88 assertions: tool contracts, failure branches, quota, no-leak guard
configs/
    default_*.yaml           one per endpoint
    api_config.example.json  credential template (real configs are gitignored and never packaged)
```

## Install

```bash
pip install -r requirements.txt
```

`rdkit` also needs its Contrib `SA_Score` module, which ships with the standard wheel.

**The toolchain is pinned exactly**, and deliberately so: RDKit supplies the objective (QED), the
SA_Score, the PAINS/Brenk/NIH catalogs and the Morgan fingerprint, so its version is part of the
measurement rather than an environment detail.

| | |
|---|---|
| Python | 3.10.19 |
| RDKit | 2025.09.6 |
| openai SDK | 2.15.0 |

This is a guarantee rather than a fragility: every molecule the benchmark depends on was recomputed
under RDKit 2026.03.4 / Python 3.13.5 and every descriptor matched to `0.000e+00`. Each run records
the interpreter and RDKit version it actually used in `run_summary.json` → `environment`, so a
re-score years from now can tell "same measurement" from "similar measurement".

### Measurement protocol (fixed)

A submitted string is parsed with sanitization on, then judged admissible **before any descriptor is
computed** — an inadmissible input is rejected, never repaired. Only then is
`rdMolStandardize.Cleanup` applied, and single-fragment plus net-zero charge are re-asserted
afterwards. Every descriptor and the fingerprint are computed on that standardized molecule.
**No tautomer canonicalization and no protonation-state adjustment is performed in either
direction**, so the form you submit is the form that is measured. Similarity is
`rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)`, Tanimoto against the identically
standardized starting molecule. The full statement lives in `tasks/q1_lipophilic/task_config.yaml`
under `reproducibility`.

## Configure an endpoint

Copy the template and fill in a key. `configs/api_config*.json` is gitignored and excluded from the
delivery zip; only the example template is shipped.

The official second-round model set is **deepseek-v4-pro / kimi-k2.6 / GLM-5.2, all at
temperature 0.0**. Only the deepseek endpoint is verified — read `_endpoint_status` in the kimi and
glm templates before spending a run on either. `default_qwen.yaml` and its template are retired
(first round only) and a qwen run must not enter a comparison table.

```bash
cp configs/api_config_deepseek.example.json configs/api_config_deepseek.json
cp configs/api_config_kimi.example.json     configs/api_config_kimi.json
cp configs/api_config_glm.example.json      configs/api_config_glm.json
python scripts/test_llm_connection.py --config configs/default_deepseek.yaml
```

## Run

```bash
python src/main.py --config configs/default_deepseek.yaml
python src/main.py --config configs/default_kimi.yaml
python src/main.py --config configs/default_glm.yaml
```

**Where runs are written.** `runtime.run_dir` is `runs`, i.e. a `runs/` directory *inside* this
package, created on first use. It used to be `../runs`, which resolves correctly only while this
public tree sits next to the grading tree inside the development repository; unpacked on its own,
that wrote a sibling directory outside the package. The grader accepts either location. Trajectories
are kept out of the public archive on purpose: a graded report quotes the reference target.

Each run writes a directory under `runs/` containing `trajectory.jsonl` (per round: every LLM turn,
every tool call and response, the submitted molecule, the computed evaluation and the returned
feedback), `conversation.md`, `llm_api_debug.jsonl`, `run_summary.json`, and `api_config_used.json`
with the key redacted.

**`run_summary.json` deliberately contains no score.** To grade a run, use the grader package:

```bash
python ../grader/scripts/score_run.py runs/<run_dir>
```

## Self-check

```bash
python tests/test_input_validation.py
python tests/test_tools.py
```

Both suites are standalone (no pytest required) and fail loudly if they run fewer assertions than
expected, so a suite that silently stops generating checks cannot pass vacuously.

## The task in one paragraph

Five rounds, one molecule per round, starting from an anilide sulfonamide whose QED is already 0.8090.
Raise QED as far as possible while keeping molecular weight in 345–385, LogP in 1.0–2.3, TPSA in
62–88, SA_Score ≤ 3.5, zero PAINS/Brenk/NIH alerts, and Morgan/Tanimoto similarity ≥ 0.40 to the
starting molecule. The similarity floor is the mechanism: it is a budget on how much of the molecule
may change, and the start sits only 12.5 Da above the molecular-weight floor while QED on this
molecule improves mainly by shedding mass — so improving edits and feasible edits are almost disjoint
sets, and progress requires several coordinated changes at once.

## Scoring in one line

`total = optimality_band(achievement)` over 0–100, where
`achievement = (best_usable_QED − start_QED) / (reference_QED − start_QED)`, clipped to [0, 1].
A design that violates any hard constraint is not usable and contributes nothing; satisfying the
constraints earns no points by itself. **The reference value and the band thresholds are held by the
grader package and are not disclosed here.**
