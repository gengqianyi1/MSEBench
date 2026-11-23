"""Run with a NEW environment's python -I; imports must resolve to that install."""
import importlib.metadata
import json
from pathlib import Path
import sys
from harbor.agents.base import BaseAgent
from agents.chembench_agent import ChemBenchAgent
from chembench_harness.protocol import resolve_case_number
import agents
import chembench_harness

assert issubclass(ChemBenchAgent, BaseAgent)
assert resolve_case_number('case 1', 1) == 1
assert importlib.metadata.version('harbor') == '0.21.0'
for module in (agents, chembench_harness):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), module.__name__
print(json.dumps({'status':'PASS', 'real_harbor_import':True, 'isolated_installed_packages':True,
                  'harbor_version':importlib.metadata.version('harbor')}))
