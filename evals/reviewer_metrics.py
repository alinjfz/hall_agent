"""Summarize reviewer activity without exporting report text."""
import argparse
import json
import sqlite3
from pathlib import Path


def reviewer_metrics(path, sample_size):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("""
            SELECT final.report_id, final.details, reports.trace_id,
                   EXISTS (SELECT 1 FROM report_activity AS edit
                           WHERE edit.report_id=final.report_id AND edit.id<final.id
                             AND edit.event='proposal_updated'
                             AND json_extract(edit.details,'$.reviewer_edits') IS NOT NULL
                             AND json_extract(edit.details,'$.reviewer_edits') <> '{}'
                             AND edit.id > COALESCE((
                                 SELECT MAX(ready.id) FROM report_activity AS ready
                                 WHERE ready.report_id=final.report_id AND ready.id<final.id
                                   AND ready.event='proposal_ready'
                             ), 0))
            FROM report_activity AS final
            JOIN reports ON reports.id=final.report_id
            WHERE final.event='review_complete' AND reports.result IS NOT NULL
              AND reports.deleted_at IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM report_activity AS newer
                  WHERE newer.report_id=final.report_id AND newer.event='review_complete' AND newer.id>final.id
              )
            ORDER BY final.id DESC
        """)
        counts = dict(approve=0, reject=0, acknowledge=0, edit=0)
        sample = []
        total = 0
        for report_id, details, trace_id, edited in rows:
            decision = json.loads(details)["decision"]
            counts[decision] += 1
            counts["edit"] += int(edited)
            total += 1
            if (edited or decision == "reject") and len(sample) < sample_size:
                sample.append(dict(report_id=report_id, trace_id=trace_id, decision=decision, edited=bool(edited)))
    return dict(completed_reviews=total, counts=counts,
                rates={key: count / total if total else None for key, count in counts.items()},
                recent_overrides=sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path)
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error("--sample-size must be non-negative")
    if not args.db.is_file():
        parser.error("Database file does not exist")
    print(json.dumps(reviewer_metrics(args.db, args.sample_size), indent=2))


if __name__ == "__main__":
    main()
