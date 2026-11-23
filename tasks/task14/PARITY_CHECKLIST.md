# Task 14 static parity checklist (old legacy runner vs Harbor multi-step)

Must be verified before `protocol_decoupled = true`.

## 1. Round protocol
- [ ] exactly 5 rounds, one candidate per round
- [ ] round N feedback is computed from round N-1 candidate only
- [ ] no early stop, no min_reward

## 2. Feedback equivalence
- [ ] computed fields identical to legacy `generate_feedback`
- [ ] `best_qed_so_far` present IFF legacy revealed it (verify against old Task 14)
- [ ] no pass/fail, no target-window restatement, no hidden reference

## 3. Message sequence equivalence
- [ ] old: system, user(r1), assistant(r1), user(feedback1), ...
- [ ] new: system, user(r1 instr), assistant/tool..., user(r2 instr), ...
- [ ] no extra instruction turn vs old feedback turn (diff the trajectories)

## 4. Scoring equivalence
- [ ] final verifier re-derives all properties from SMILES
- [ ] best usable candidate = feasible AND QED strictly above start
- [ ] ignores best_so_far.json / feedback.json / agent-reported values

## 5. Resume
- [ ] protocol_state.json message history == legacy messages list
- [ ] later: native --resume-trajectory equivalent
