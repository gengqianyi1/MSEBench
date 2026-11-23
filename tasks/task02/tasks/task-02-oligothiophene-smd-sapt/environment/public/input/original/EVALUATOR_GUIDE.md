# Evaluator-only material

The wrapper keeps evaluator-only assets in `private_data/`, outside this public
`input/original` tree. `native_runner.py` mounts only this public tree into the
Agent container. After the container exits, the harness invokes
`private_grader/grader.py` and supplies `private_data_dir` in the protected
grader process. Never mount or copy `private_data/` into an Agent workspace.
