"""Protocol plugin registry: protocol name -> plugin module (or None = builtin).

Builtins are implemented as methods on msebench.agent.unified.MSEBenchAgent and
dispatched by `cfg["protocol_mode"]`; plugins implement `run(agent, instruction,
environment, context, cfg)` and are loaded via MSEBENCH_PROTOCOL_MODULE.
"""
PROTOCOLS = {
    "workspace_sidecar_case": "msebench.protocols.workspace_sidecar",
    "hidden_suite_case": None,
    "adaptive_hidden_suite_case": None,
    "task03_design_round": None,
    "private_response_round": None,
    "candidate_only_round": None,
    "task15_diagnostic_round": None,
    "textqa": None,
}


def plugin_module(protocol):
    return PROTOCOLS.get(protocol)


def is_builtin(protocol):
    return protocol in PROTOCOLS and PROTOCOLS[protocol] is None
