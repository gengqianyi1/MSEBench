# Public reaction validator

Run `python input/tools/reaction_validator.py route.json`. Input may be a route array or an object with top-level `route`. Output is JSON with per-step normalized reaction name, generated products, declared-product match, unavailable reactants, heavy-atom conservation, atom economy, and error status.

Runtime: the pinned offline RDKit environment in `environment.yml`. The tool has no network dependency, private target oracle, or reference route. The private grader imports this exact module and adds final-target scoring.
