"""Controlled-tool template. Tool names, schemas and budgets require author confirmation."""
from chembench_harness import TaskPlugin


class ControlledToolPlugin(TaskPlugin):
    def initialize(self): return {"tool_calls": {}}
    def validate_action(self, action, state):
        # AUTHOR TODO: allowlist action/tool names and validate arguments against JSON schema.
        raise NotImplementedError
    def execute_action(self, action, state, workspace):
        # AUTHOR TODO: call a protected implementation; do not expose checkpoints/private cases.
        raise NotImplementedError
    def public_feedback(self, result, state):
        # AUTHOR TODO: redact evaluator-only fields.
        raise NotImplementedError
    def is_terminal(self, state):
        # AUTHOR TODO: declare the exact terminal condition.
        raise NotImplementedError
    def materialize_submission(self, state, workspace):
        # AUTHOR TODO: preserve action/tool audit evidence.
        raise NotImplementedError
