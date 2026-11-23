# Building-block-constrained synthesis route validation

Plan a forward route to the target below using only the listed purchasable leaves and named reaction library.

Target: `CC(=O)N(CCOC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1)Cc1ccccc1`

Read `input/TASK_SPEC.md` for the 25-item inventory, eight versioned reaction SMARTS, step schema, exact thresholds, equivalence rules and scoring. During planning, write a tentative route JSON and run `python input/tools/reaction_validator.py <route.json>`. The public validator returns per-step reaction firing, declared-product matching, reactant availability, heavy-atom conservation, atom economy, and actionable errors without revealing the private reference route. Revise the route using that feedback.

Return top-level key `route`, an ordered array of 2-6 step objects (`reaction`, `reactants`, `product`), inside the response's `final_answer`; the Harness saves canonical `final_answer.json`. You may also write identical `route.json` as an optional audit artifact. <!-- 2026-09-27 fix: made the grader's file priority explicit; the grader previously read route.json first, letting a draft override the canonical answer. --> The private grader scores `final_answer.json` and consults `route.json` only as a fallback when `final_answer.json` is absent, so a draft `route.json` never overrides the canonical answer. Reaction names are normalized by case-folding and replacing spaces/hyphens with underscores; chemistry remains governed by the same SMARTS.

The private RDKit grader imports the same public reaction-contract module, independently checks the full route and verifies the exact final target. Candidate-reported scores are ignored. <!-- 2026-09-27 fix: scoring normalization disclosed; the 2-6 step legality range previously read as if shorter routes could earn full progress credit. --> Scoring note: the 35-point verified-step-progress component is normalized against the reference route length (6 steps), so a legal route shorter than the reference earns proportionally fewer progress points — using fewer than 6 steps grants no extra progress credit. This is a single-instance deterministic chemistry-compliance/route-validation task. Until it is expanded to a frozen multi-target pool, it is excluded from the Agent leaderboard and cross-task capability mean and must not be interpreted as equivalent to open-ended scientific search.

---

## MSEBench Harbor execution contract (protocol v2.1)

The evaluated model is supplied by the Harbor Agent layer; this task must not call the evaluated LLM directly. Work only inside `/workspace`. Required legacy artifacts: `final_answer.json`.

<!-- 2026-09-27 fix: budget disclosure added; the instruction previously omitted the runtime budget that .chembench_agent.json / harness_contract.yaml already enforce. -->

Budget limits: at most 5 agent rounds and 15 monitored tool calls (see `.chembench_agent.json` / `harness_contract.yaml`; 600 s per tool call, no automatic retries). The harness saves `final_answer.json` only when the model returns `final_answer`. Exhausting the round/tool budget without producing a scorable answer — `final_answer.json`, or an identical `route.json` only via the fallback described above when `final_answer.json` is absent — yields 0 points.
