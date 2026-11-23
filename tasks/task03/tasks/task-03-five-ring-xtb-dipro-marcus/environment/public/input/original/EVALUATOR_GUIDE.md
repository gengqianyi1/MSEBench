# Evaluator-only material

This task's evaluator-only assets are deliberately outside this task root, at `../../evaluator_private/original_tasks/03_five_ring_xtb_dipro_marcus/`. Do not mount that directory into the agent container, do not copy it into `runs/<run_id>/workspace`, and do not expose it in the prompt. Run the candidate with only this task directory mounted. After the run finishes, invoke the scorer from the evaluator host and pass the required asset path explicitly.
