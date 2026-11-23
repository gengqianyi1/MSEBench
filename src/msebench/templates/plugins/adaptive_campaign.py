"""Adaptive-campaign template. Selection state and compute budgets are framework-owned."""
from msebench_harness import TaskPlugin


class AdaptiveCampaignPlugin(TaskPlugin):
    def initialize(self): return {"observations": [], "selected": []}
    def validate_action(self, action, state):
        # AUTHOR TODO: validate the next experiment and reject repeats/out-of-grid choices.
        raise NotImplementedError
    def execute_action(self, action, state, workspace):
        # AUTHOR TODO: run the approved scientific executable in the protected environment.
        raise NotImplementedError
    def public_feedback(self, result, state):
        # AUTHOR TODO: expose the exact legacy observation, including approved failure information.
        raise NotImplementedError
    def is_terminal(self, state):
        # AUTHOR TODO: use the original compute/experiment budget.
        raise NotImplementedError
    def materialize_submission(self, state, workspace):
        # AUTHOR TODO: emit the campaign trace required by the independent grader.
        raise NotImplementedError
