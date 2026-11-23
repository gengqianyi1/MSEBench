# Private evaluator boundary

The RDKit/xTB/DIPRO evaluator and its implementation are organizer-private.
During each design round, the Agent submits the documented candidate JSON to
the Task 03 sidecar and receives only the feedback fields declared in
`question.md`.  No evaluator module is available in the agent image.
