from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from unittest.mock import patch

from langsmith import Client, tracing_context
from streamlit.testing.v1 import AppTest

from tests.scripted_model import scripted, wait_for_triage


from unihelp import models, workflow
from unihelp.store import Store

def test_chase_review_and_replay(monkeypatch):
    with TemporaryDirectory() as d:
        db = Path(d) / "demo.sqlite3"
        monkeypatch.setenv("UNIHELP_DB", str(db))
        monkeypatch.setenv("UNIHELP_PROVIDER", "openai")
        store = Store(db)
        store.seed()
        with store.transaction(write=True) as conn:
            conn.execute("UPDATE reports SET triage_error='Excluded from this smoke test' WHERE id != 1")
        roots = []
        client = Client(api_url="http://127.0.0.1:1", auto_batch_tracing=False)
        review = workflow.review
        original_triage = workflow._triage
        original_review = workflow._review

        def capture(function):
            def call(*args, langsmith_extra, **kwargs):
                with tracing_context(enabled="local", client=client):
                    return function(*args, **kwargs, langsmith_extra=langsmith_extra | {"on_end": roots.append})
            return call

        with patch.object(Client, "create_run"), patch.object(Client, "update_run"), \
             patch.object(models, "setup_issue", return_value=None), \
             patch.object(workflow, "triage_model", return_value=scripted(
            "Juniper Court / Flat 12 kitchen", "escalation",
            {"intent": "chase", "candidate_ids": [101], "evidence": "ceiling light"},
        )), \
             patch.object(workflow, "_triage", capture(original_triage)), \
             patch.object(workflow, "_review", capture(original_review)):
            ui = AppTest.from_file(str(Path("app.py").resolve())).run(timeout=30)
            wait_for_triage(ui, store, 1)
            assert not ui.exception, ui.exception
            assert not ui.error and not ui.exception, (ui.error, ui.exception)
            next(b for b in ui.button if b.label == "Approve escalation").click().run(timeout=30)
            assert not ui.error and not ui.exception
            report = store.report(1)
            assert report["result"]["action"] == "escalated"
            with ThreadPoolExecutor(max_workers=2) as pool:
                receipts = list(pool.map(lambda _: review(store, 1, report["version"], "approve"), range(2)))
            assert receipts == [report["result"]] * 2
            assert store.jobs()[0]["escalated_at"] == report["result"]["at"]
            assert sum(bool(job["escalated_at"]) for job in store.jobs()) == 1

        def walk(run):
            yield run
            for child in run.child_runs:
                yield from walk(child)

        nodes = list(walk(roots[0]))
        print([(node.name, node.run_type) for node in nodes])
        assert {"load_report_and_context", "retrieve_handbook", "search_handbook", "apply_policy", "await_review"} <= {
            node.name for node in nodes}
        search = next(node for node in nodes if node.name == "search_handbook")
        assert search.run_type == "retriever" and "ceiling light" in str(search.inputs)
        assert roots[1].outputs["action"] == "escalated", roots[1].outputs
        assert roots[1].metadata["triage_trace_id"] == str(roots[0].id)
        assert "apply_decision" in {node.name for node in walk(roots[1])}
        print("UI chase-up, replay, SQLite timestamp, and local LangSmith trace verified.")
