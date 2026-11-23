# Descriptor-surrogate electrolyte solvent design

Propose one neutral, single-fragment CHOF molecule as SMILES. This benchmark is a deterministic **task-internal RDKit descriptor optimization**. Its electrochemical-window value is a rule surrogate computed from O/F counts, rotatable bonds and logP. It is not an orbital calculation and must not be interpreted as evidence of real high-temperature fast-charge performance, interfacial stability, lithium plating behaviour, or overall electrolyte quality.

The fixed rules are documented in `input/MODEL_CARD.md`. The formal score never switches to xTB through environment variables; xTB studies are a separate, non-scoring research activity.

All rules: valid RDKit molecule; elements only C/H/O/F; formal charge 0; one fragment; no radicals; 7-16 heavy atoms; MW below 250; descriptor surrogate strictly between -7.20 and -6.35; MW 155-210; logP -0.2 to 0.9; 4-5 rotatable bonds; 4-5 oxygen atoms; HBD 0; task SA score below 2.8; at most 2 fluorines; no rings or aromatic atoms; no epoxide, peroxide, acid halide, anhydride, or sp3 carbon with three or more oxygen neighbours. A private versioned reserved set is a zero-score hard gate.

Return `smiles`, `name`, and `design_rationale` inside the response's `final_answer` object; the Harness saves it as canonical `final_answer.json`. You may also write an identical `candidate.json`, which is collected as an optional audit artifact. <!-- 2026-09-27 fix: made the grader's file priority explicit; the grader previously read candidate.json first, letting a draft override the canonical answer. --> The private grader scores `final_answer.json` and consults `candidate.json` only as a fallback when `final_answer.json` is absent, so a draft `candidate.json` never overrides the canonical answer. The private grader canonicalizes the SMILES, independently computes all descriptors and SMARTS/atom-neighbour checks, and ignores candidate-reported properties or scores.

Interpretation boundary: the score reports performance only under this frozen RDKit descriptor surrogate. It is not evidence of a real HOMO level, SEI behaviour, fast-charge performance, or electrolyte quality. Because a no-tool direct-answer baseline has reached full score, v2.0.2 is classified as a static cheminformatics-compliance task and is excluded from the Agent leaderboard and cross-task capability mean.

---

## MSEBench Harbor execution contract (protocol v2.1)

The evaluated model is supplied by the Harbor Agent layer; this task must not call the evaluated LLM directly. Work only inside `/workspace`. Required legacy artifacts: `final_answer.json`.

<!-- 2026-09-27 fix: budget disclosure added; the instruction previously omitted the runtime budget that .chembench_agent.json / harness_contract.yaml already enforce. -->

Budget limits: at most 6 agent rounds and 12 monitored tool calls (see `.chembench_agent.json` / `harness_contract.yaml`; 300 s per tool call, no automatic retries). The harness saves `final_answer.json` only when the model returns `final_answer`. Exhausting the round/tool budget without producing a scorable answer — `final_answer.json`, or an identical `candidate.json` only via the fallback described above when `final_answer.json` is absent — yields 0 points.
