"""Sweep the handbook score cut-off without a model, using report text as the query."""

import json
from pathlib import Path

from halldesk.retrieval import handbook_index

CASES = json.loads((Path(__file__).with_name("dataset.json")).read_text())


def sweep(thresholds=(0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
    _, index = handbook_index()
    cases = [case for case in CASES if case.get("policy_sections") and case["slice"] != "non-English"]
    rows = []
    for threshold in thresholds:
        found = [({key for key, _ in index.search(case["inputs"]["message"], 3, min_score=threshold)},
                  set(case["policy_sections"])) for case in cases]
        rows.append({
            "min_score": threshold,
            "recall_at_3": round(sum(len(got & expected) / len(expected) for got, expected in found) / len(found), 3),
            "off_target_per_query": round(sum(len(got - expected) for got, expected in found) / len(found), 2),
            "empty": sum(not got for got, _ in found), "queries": len(found),
        })
    return rows


if __name__ == "__main__":
    for row in sweep():
        print(json.dumps(row))
