"""Evaluate the safety regex on synthetic reports without calling a model."""
import json
from pathlib import Path

from halldesk.safety import safety_signals

if __name__ == "__main__":
    cases = json.loads(Path(__file__).with_name("safety_cases.json").read_text())
    misses = false_alarms = 0
    for case in cases:
        signals = safety_signals(case["message"])
        if bool(signals) != case["hazard"]:
            misses += int(case["hazard"])
            false_alarms += int(not case["hazard"])
            label = "MISS" if case["hazard"] else "FALSE ALARM"
            print(f"{label}: {case['message']} ({', '.join(sorted(signals)) or 'no signal'})")
    print(f"{len(cases)} cases: {misses} misses, {false_alarms} false alarms")
    raise SystemExit(int(bool(misses or false_alarms)))
