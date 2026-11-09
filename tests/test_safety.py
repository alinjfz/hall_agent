import json
from pathlib import Path

import pytest

from halldesk.safety import safety_signals


def safety_examples():
    for row in json.loads(Path(__file__).with_name("safety_examples.json").read_text()):
        yield pytest.param(row["message"], row["hazard"], id=row["message"])


@pytest.mark.parametrize(("message", "hazard"), list(safety_examples()))
def test_intake_safety_examples(message, hazard):
    assert bool(safety_signals(message)) == hazard
