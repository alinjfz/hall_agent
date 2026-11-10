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


@pytest.mark.parametrize(("message", "signals"), [
    ("A picture on the wall shows flames from the old drill.", set()),
    ("What should I do if I smell gas in the future?", set()),
    ("The diagram reads 'exposed wires' as a warning.", set()),
    ("A video shows flames. My oven has ignited.", {"fire"}),
    ("What should I do if I smell gas right now?", {"gas"}),
    ("A manual says 'exposed wires' as a warning, and there are exposed wires in my room.",
     {"building_hazard"}),
])
def test_reference_does_not_hide_a_current_hazard(message, signals):
    assert safety_signals(message) == signals
