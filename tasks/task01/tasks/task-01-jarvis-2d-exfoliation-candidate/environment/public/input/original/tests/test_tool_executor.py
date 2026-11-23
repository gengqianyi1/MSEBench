import pytest

from src.tool_executor import safety_check


def test_agent_python_cannot_import_chgnet_directly():
    with pytest.raises(ValueError, match="forbidden import"):
        safety_check("import chgnet", max_script_size_bytes=1000)


def test_regular_scientific_import_remains_allowed():
    safety_check("import numpy as np", max_script_size_bytes=1000)
