"""Measure handbook retrieval without a chat model.

Queries are built by the production `handbook_query`. "message" uses the report text
alone; "facts" adds the intent and category a correct interpretation would supply,
taken from each case's expected outcome. Lexical search needs no services. Pass
--hybrid with HALL_EMBEDDING_MODEL set to sweep the embedding similarity floor.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from halldesk import retrieval
from halldesk.context import REPAIR_CATEGORIES, handbook_query

CASES = json.loads((Path(__file__).with_name("dataset.json")).read_text())
INTENTS = {"escalate": "chase", "draft": "chase", "job_question": "chase", "create": "new",
           "lighting_question": "new", "plumbing_question": "new", "split": "multiple",
           "access_question": "access", "answer": "question"}


def query_for(case, mode):
    message, outputs = case["inputs"]["message"], case["outputs"]
    if mode == "message":
        return handbook_query(message)
    route = outputs.get("route")
    category = outputs.get("category") or next(
        (name for name in REPAIR_CATEGORIES if route == f"{name}_question"), None)
    return handbook_query(message, intent=INTENTS.get(route), category=category)


def score(cases, mode, search):
    found = [([document.id for document in search(query_for(case, mode))], set(case["policy_sections"]))
             for case in cases]
    labelled = [(got, expected) for got, expected in found if expected]
    unanswerable = [got for got, expected in found if not expected]
    ranks = [next((rank for rank, key in enumerate(got, start=1) if key in expected), None)
             for got, expected in labelled]
    return {
        "queries": len(labelled),
        "recall_at_3": round(sum(len(set(got) & expected) / len(expected) for got, expected in labelled)
                             / len(labelled), 3),
        "hit_at_1": round(sum(rank == 1 for rank in ranks) / len(labelled), 3),
        "mrr": round(sum(1 / rank for rank in ranks if rank) / len(labelled), 3),
        "off_target_per_query": round(sum(len(set(got) - expected) for got, expected in labelled) / len(labelled), 2),
        "empty": sum(not got for got, _ in labelled),
        # Questions the handbook cannot answer should retrieve nothing.
        "unanswerable_left_empty": f"{sum(not got for got in unanswerable)}/{len(unanswerable)}",
    }


def sweep(hybrid=False, thresholds=None):
    english = [case for case in CASES if case["slice"] != "non-English"
               and (case.get("policy_sections") or "question" in case["policy_topics"])]
    other = [case for case in CASES if case["slice"] == "non-English" and case.get("policy_sections")]
    name, default = ("MIN_SIMILARITY", retrieval.MIN_SIMILARITY) if hybrid else ("MIN_SCORE", retrieval.MIN_SCORE)
    rows = []
    try:
        for threshold in thresholds or ((0.1, 0.2, 0.3, 0.4, 0.5, 0.6) if hybrid else (0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
            setattr(retrieval, name, threshold)
            retriever = retrieval.handbook_retriever()
            if hybrid and retriever.metadata["retrieval_mode"] != "hybrid":
                raise SystemExit("Set HALL_EMBEDDING_MODEL and start the embedding service for --hybrid.")
            for label, cases in (("english", english), ("non-English", other)):
                for mode in ("message", "facts"):
                    rows.append({name.lower(): threshold, "slice": label, "query": mode,
                                 **score(cases, mode, retriever.invoke)})
    finally:
        setattr(retrieval, name, default)
    return rows


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", action="store_true", help="Sweep the embedding floor instead of the BM25 floor")
    for row in sweep(parser.parse_args().hybrid):
        print(json.dumps(row))
