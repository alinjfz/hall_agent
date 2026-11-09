"""Export explicitly approved holdouts and guidance for a reproducible experiment."""

import argparse
import json
from pathlib import Path

from halldesk.store import Store


def export_cases(path):
    store = Store(path)
    cases = []
    for row in store.eval_cases("holdout"):
        snapshot = row["snapshot"]
        outputs = row["expected"].copy()
        outputs["route"] = outputs.pop("kind")
        if "candidates" in outputs:
            outputs["candidates"] = sorted(outputs["candidates"])
        inputs = {"message": snapshot["message"], "location": snapshot["location"],
                  "open_jobs": snapshot["jobs"], "clock": snapshot["at"]}
        if snapshot.get("job_reports"):
            inputs["job_reports"] = snapshot["job_reports"]
        cases.append({"id": f"curated-{row['id']}", "slice": "curated", "inputs": inputs,
                      "outputs": outputs, "policy_topics": [], "policy_revisions": row["policy_revisions"]})
    return {"cases": cases, "guidance": [row for row in store.eval_cases("guidance") if row["interpretation"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error("Workspace database does not exist.")
    export = export_cases(args.database)
    if not export["cases"]:
        parser.error("Approve at least one holdout before exporting.")
    args.output.write_text(json.dumps(export, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
