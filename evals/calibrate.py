"""Compare the reply judges with independently reviewed dev or test labels."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from evals.judge import RUBRIC_VERSION, RUBRICS, judge_model_id, judge_reply


def calibration(rows):
    results = {}
    for criterion in RUBRICS:
        selected = [row for row in rows if row["criterion"] == criterion]
        bad = [row for row in selected if not row["label"]]
        good = [row for row in selected if row["label"]]
        results[criterion] = {
            "bad_replies_caught": sum(not row["verdict"] for row in bad) / len(bad) if bad else None,
            "good_replies_passed": sum(row["verdict"] for row in good) / len(good) if good else None,
            "bad_total": len(bad), "good_total": len(good),
            "disagreements": [row for row in selected if row["label"] != row["verdict"]],
        }
    return results


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [row for row in json.loads(args.labels.read_text()) if row["split"] == args.split]
    if not cases:
        parser.error("No independently labelled replies in that split.")
    try:
        model = judge_model_id()
    except ValueError as exc:
        parser.error(str(exc))
    rows = []
    for case in cases:
        if set(case["labels"]) != set(RUBRICS) or any(type(label) is not bool for label in case["labels"].values()):
            parser.error(f"{case['id']} needs one Boolean human label per criterion.")
        for criterion, label in case["labels"].items():
            outcome = judge_reply(criterion, report=case["report"], reply=case["reply"],
                                  sections=case["sections"], job=case.get("job"))
            rows.append({"id": case["id"], "criterion": criterion, "label": label, **outcome.model_dump()})
    output = json.dumps({"judge_provider": "openai", "judge_model": model, "judge_rubric_version": RUBRIC_VERSION,
                         "split": args.split,
                         **calibration(rows)}, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
