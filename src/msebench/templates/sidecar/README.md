# Protected protocol sidecar template

Use this boundary when hidden cases, response models, licensed programs, or private evaluator state
must not appear in the Agent filesystem.

## Framework-owned — do not modify

- `chembench_harness.gateway.ProtectedToolGateway`
- request envelope `{"tool":"name","arguments":{...}}`
- quota persistence before validation and handler execution
- fail-closed response-field allowlisting and standard error codes
- framework ownership of `/workspace/protocol_state.json`

Changing these creates a new harness protocol version and requires every task to be revalidated.

## Task-author editable

- tool names, argument schemas, integer budgets and timeouts
- public response-field allowlists
- handler scientific implementations
- case transitions and termination policy

Every change requires boundary, invalid-request, crash/timeout, leakage, Oracle and parity tests.

## Task-author required additions

- private handler and private-image build recipe
- field-by-field disclosure justification for every public response field
- one valid Oracle and at least three invalid/adversarial fixtures
- immutable private-image digest, never `latest`
- same-model legacy-versus-sidecar parity tolerance and evidence

The participant package contains only the public API contract and pinned image reference. Private
source, hidden cases, model parameters, labels and calibration replays stay in the organizer package
or private registry.
