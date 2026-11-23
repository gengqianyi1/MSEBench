# PyBaMM state-continuous fast-charge optimization

Submit one CC-CV fast-charge protocol and one set of electrode parameters. The private grader independently runs one continuous PyBaMM experiment: charge at CC, hold at CV, rest, then discharge. Model and parameter set are frozen to SPMe and Chen2020.

Return a `candidate` object inside the response's `final_answer`; the Harness saves canonical `final_answer.json`. It must contain these finite numeric fields: `cc_c_rate`, `cc_until_voltage_V`, `cv_voltage_V`, `cv_until_current_C`, `negative_electrode_thickness_um`, `positive_electrode_thickness_um`, `electrode_porosity`, and `particle_radius_um`; `name` and `rationale` are optional. You may also write identical `candidate.json` as an optional audit artifact. <!-- 2026-09-27 fix: made the grader's file priority explicit; the grader previously read candidate.json first, letting a draft override the canonical answer. --> The private grader scores `final_answer.json` and consults `candidate.json` only as a fallback when `final_answer.json` is absent, so a draft `candidate.json` never overrides the canonical answer. The canonical envelope is:

```json
{"candidate":{"cc_c_rate":2.0,"cc_until_voltage_V":4.1,"cv_voltage_V":4.1,"cv_until_current_C":0.15,"negative_electrode_thickness_um":100,"positive_electrode_thickness_um":90,"electrode_porosity":0.35,"particle_radius_um":4.5}}
```

For backward compatibility with archived submissions, the same eight fields may also appear at the top level. Both shapes are defined by `input/schemas/candidate.schema.json` and receive identical scientific scoring.

Read `input/MODEL_CARD.md` for every public range and the exact score contract. In particular, the 1400 mol m-3 minimum is a benchmark-internal negative-particle surface-concentration constraint under this exact model and parameter set. It is not a validated lithium-plating criterion and must not be described as one.

---

## MSEBench Harbor execution contract (protocol v2.1)

The evaluated model is supplied by the Harbor Agent layer; this task must not call the evaluated LLM directly. Work only inside `/workspace`. Required legacy artifacts: `final_answer.json`.

<!-- 2026-09-27 fix: budget disclosure added; the instruction previously omitted the runtime budget that .chembench_agent.json / harness_contract.yaml already enforce. -->

Budget limits: at most 6 agent rounds and 12 monitored tool calls (see `.chembench_agent.json` / `harness_contract.yaml`; 900 s per tool call, no automatic retries). The harness saves `final_answer.json` only when the model returns `final_answer`. Exhausting the round/tool budget without producing a scorable answer — `final_answer.json`, or an identical `candidate.json` only via the fallback described above when `final_answer.json` is absent — yields 0 points.
