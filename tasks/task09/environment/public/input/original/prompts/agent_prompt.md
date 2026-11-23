You are reconstructing the gas-phase GFN2-xTB torsional profile for the frozen
2-fluorobiphenyl case. Respond with exactly one JSON action and no prose,
Markdown, code fence, or additional keys.

Only two actions exist:

1. `run_scan_batch` with `angles_deg`, `finalize`, `confidence`, and a concise
   `rationale`.
2. `inspect_failed_attempt` for an angle that the runtime reports as failed
   after both automatic native attempts.

All angles must be distinct integer multiples of 5 in `[0,360)` and must not
have appeared in a prior accepted batch. You choose every angle; the runtime
never chooses for you.

- Phase 0: submit exactly six new angles with `finalize=false` and
  `confidence=null`.
- Phase 1: after the first automatic profile analysis, submit two or three new
  angles with `finalize=false` and `confidence=null`.
- Phase 2: after the second analysis, submit two or three new angles with
  `finalize=true` and confidence `low`, `medium`, or `high`.
- Recovery is available only when the runtime explicitly reports insufficient
  valid points after automatic retry. It accepts one or two replacement angles
  without exceeding twelve total unique angles.

Use the compact analysis to refine large gaps, high-curvature or
high-disagreement intervals, and tentative extrema. Eligibility requires
10–12 valid unique angles, two feedback rounds, and three automatic profile
fits. Finalization is terminal: the runtime binds the selected deterministic
profile and creates the closed final envelope without another provider turn.

The rationale must be 1–400 characters. Do not request or speculate about the
oracle, filesystem, source, evaluator, credentials, arbitrary paths, shell,
Python, network access, hidden reasoning, scores, hashes, or full provenance.
