"""Multi-round template. Core round counters/budgets are framework-owned."""
from msebench_harness import TaskPlugin


class MultiRoundPlugin(TaskPlugin):
    MAX_ROUNDS = 5  # AUTHOR EDITABLE: must match the original protocol.
    def initialize(self): return {"best": None}
    def validate_action(self, action, state):
        # AUTHOR TODO: validate candidate JSON and edit constraints; never trust self-reported metrics.
        raise NotImplementedError
    def execute_action(self, action, state, workspace):
        # AUTHOR TODO: independently evaluate the candidate using the approved scientific ruler.
        raise NotImplementedError
    def public_feedback(self, result, state):
        # AUTHOR TODO: return only author-approved public fields; never pass hidden thresholds/optima.
        raise NotImplementedError
    def is_terminal(self, state): return state.current_round >= self.MAX_ROUNDS
    def materialize_submission(self, state, workspace):
        # AUTHOR TODO: write trajectory.jsonl/run_summary.json without grader-side secrets.
        raise NotImplementedError
