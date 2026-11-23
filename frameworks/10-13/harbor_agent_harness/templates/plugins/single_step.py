"""Single-step template. Authors replace TODO methods only; do not change method signatures."""
from chembench_harness import ProtocolState, TaskPlugin


class SingleStepPlugin(TaskPlugin):
    def initialize(self): return {}
    def validate_action(self, action, state):
        # AUTHOR TODO: validate the declared artifact/action schema.
        raise NotImplementedError
    def execute_action(self, action, state, workspace):
        # AUTHOR TODO: materialize only public, independently checkable results.
        raise NotImplementedError
    def public_feedback(self, result, state): return result
    def is_terminal(self, state): return state.current_round >= 1
    def materialize_submission(self, state, workspace):
        # AUTHOR TODO: create the exact legacy-compatible scoring surface.
        raise NotImplementedError
