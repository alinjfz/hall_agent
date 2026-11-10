import json
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from unihelp import models, workflow
from unihelp.store import LONDON, Store
from tests.scripted_model import scripted, wait_for_triage


def click(ui, label):
    next(button for button in ui.button if button.label == label).click().run(timeout=30)
    assert not ui.error and not ui.exception, (ui.error, ui.exception)


def edit_job_status(ui, job_id, status):
    # AppTest cannot set data editor cells directly. Send the browser's widget payload.
    editor = ui.data_editor[0]
    row_index = next(i for i, value in enumerate(editor.value["ID"]) if value == job_id)
    state = ui._tree.get_widget_states()
    state.widgets.add(id=editor.proto.id, string_value=json.dumps({
        "edited_rows": {row_index: {"Status": status}},
    }))
    ui._run(state, timeout=30)
    assert not ui.error and not ui.exception, (ui.error, ui.exception)
    assert any(toast.value == "Job status saved." for toast in ui.toast)
    assert ui.data_editor[0].proto.id == editor.proto.id


def test_inbox_review_and_job_lifecycle():
    with TemporaryDirectory() as directory:
        db = Path(directory) / "inbox.sqlite3"
        store = Store(db)
        store.seed()
        with store.transaction(write=True) as conn:
            conn.execute("UPDATE reports SET triage_error='Excluded from this smoke test' WHERE id NOT IN (2,3)")
        repair = {
            "intent": "new", "candidate_ids": [], "evidence": "ceiling light", "fitting": "ceiling light",
            "other_lighting_works": True, "lighting_evidence": "The wall light in the kitchen still works.",
        }
        chase = {"intent": "chase", "candidate_ids": [102, 103], "evidence": "light repair I reported"}
        responses = [
            scripted(store.report(2)["location"], "lighting", repair),
            scripted(store.report(3)["location"], "escalation", chase),
            scripted("Juniper Court / Flat 21 study", "lighting", {
                "intent": "new", "candidate_ids": [], "evidence": "desk light", "fitting": "desk light",
            }),
        ]
        original_triage = workflow.triage
        def scripted_triage(store, report_id, **kwargs):
            return original_triage(store, report_id, model=responses.pop(0), **kwargs)

        with patch.dict("os.environ", {
            "UNIHELP_DB": str(db), "LANGSMITH_TRACING": "false", "UNIHELP_PROVIDER": "openai",
            "UNIHELP_ENABLE_QUALITY_CHECKS": "true",
        }), \
             patch.object(models, "setup_issue", return_value=None), \
             patch.object(workflow, "triage", side_effect=scripted_triage) as model:
            ui = AppTest.from_file(str(Path("app.py").resolve())).run(timeout=30)
            assert not ui.exception, ui.exception
            wait_for_triage(ui, store, 2, 3)
            assert model.call_count == 2
            assert store.report(2)["proposal"]["kind"] == "create"
            assert store.report(3)["proposal"]["kind"] == "job_question"
            assert len(store.jobs()) == 3
            ui.radio(key="message-list").set_value(2).run(timeout=30)
            ui.multiselect[0].set_value(["ceiling_light"]).run(timeout=30)
            due_date = store.clock().astimezone(LONDON).date() + timedelta(days=3)
            next(widget for widget in ui.date_input if widget.label == "Deadline override").set_value(due_date).run(timeout=30)
            assert next(button for button in ui.button if button.label == "Approve new job").disabled
            click(ui, "Save review edits")
            click(ui, "Approve new job")
            receipt = store.report(2)["result"]
            assert receipt["action"] == "created_job"
            assert receipt["reviewer_edits"] == {"labels": ["ceiling_light"], "due_date": due_date.isoformat()}
            job = next(job for job in store.jobs() if job["created_from"] == 2)
            assert job["labels"] == ["ceiling_light"]
            assert job["deadline"].startswith(f"{due_date}T23:59:59")

            ui.radio(key="message-list").set_value(3).run(timeout=30)
            assert {"Select job #102", "Select job #103"} <= {button.label for button in ui.button}
            next(widget for widget in ui.text_area if widget.label == "Resident reply draft").set_value(
                "Which light on the landing did you report?"
            )
            click(ui, "Approve and post reply")
            assert store.activity(3)[0]["details"]["body"] == "Which light on the landing did you report?"
            assert "Approve and post reply" not in {button.label for button in ui.button}
            next(widget for widget in ui.radio if widget.label == "From").set_value("resident")
            next(widget for widget in ui.text_area if widget.label == "Message").set_value("The wall light.")
            click(ui, "Add to conversation")
            assert store.activity(3)[0]["details"]["author"] == "resident"
            click(ui, "Select job #103")
            assert store.report(3)["proposal"]["kind"] == "draft"
            click(ui, "Decline draft")
            assert store.activity(3)[0]["event"] == "reply_declined"
            click(ui, "Mark report as handled")
            assert store.report(3)["result"]["action"] == "draft"
            assert all(job["escalated_at"] is None for job in store.jobs())

            report = store.report(7)
            safety_model = scripted(report["location"], "lighting", {"intent": "manual"}, safety={
                "assessment": "current", "hazard": "building_hazard", "evidence": "sparks coming from the fitting",
            })
            original_triage(store, 7, model=safety_model)
            ui.radio(key="message-list").set_value(7).run(timeout=30)
            click(ui, "Record technician handoff")
            assert store.report(7)["result"]["action"] == "technician_handoff"
            assert store.activity(7)[0]["details"]["action"] == "technician_handoff"
            assert any(receipt.value == "Handed over to technicians" for receipt in ui.success)
            assert len(store.jobs()) == 4

            click(ui, "New report")
            location = ui.selectbox(key="compose-location")
            assert "Flat 12 kitchen" in location.options
            location.set_value("Flat 21 study")
            ui.text_input(key="compose-subject").set_value("  Study light  ")
            ui.text_area(key="compose-message").set_value("  The desk light broke.\r\n\r\nPlease check.  ")
            click(ui, "Add to inbox")
            wait_for_triage(ui, store, 9)
            assert model.call_count == 3, (ui.error, store.reports()[-1])
            assert store.report(9)["proposal"]["kind"] == "lighting_question"
            assert store.report(9)["subject"] == "Study light"
            assert store.report(9)["message"] == "The desk light broke.\n\nPlease check."
            assert ui.radio(key="message-list").value == 9

            click(ui, "Save as quality check")
            ui.selectbox(key="expected-9").set_value("lighting_question").run(timeout=30)
            click(ui, "Save draft")
            quality = next(row for row in store.eval_cases() if row["name"] == "Study light")
            assert quality["status"] == "draft"
            ui.button_group(key="workspace").set_value("Quality checks").run(timeout=30)
            ui.button(key=f"eval-{quality['id']}-holdout").click().run(timeout=30)
            assert store.eval_cases("holdout")[0]["expected"]["kind"] == "lighting_question"

            ui.button_group(key="workspace").set_value("Reports").run(timeout=30)
            with patch.dict("os.environ", {"UNIHELP_BACKGROUND_WORKER": "true"}):
                similar_id = store.add_report("Flat 1", "Another ceiling light", "The ceiling light has stopped working.")
                original = store.report(2)["proposal"]
                store.save_proposal(store.report(similar_id), {
                    key: original[key] for key in ("kind", "rule", "fitting", "labels", "category", "priority", "days")
                })
                ui.run(timeout=30)
                ui.radio(key="message-list").set_value(similar_id).run(timeout=30)
                assert any(f"Previous review for the same fitting · Job #{job['id']}" in entry.value for entry in ui.caption)
                ui.button_group(key="mailboxes").set_value("Failed").run(timeout=30)
                assert ui.radio(key="message-list").value == 1
                click(ui, "Retry triage")
                assert ui.radio(key="message-list").value == 1
                assert store.report(1)["triage_error"] is None
                assert store.activity(1)[0]["event"] == "requeued"
                # Keep the queued test case out of the scripted model's remaining runs.
                with store.transaction(write=True) as conn:
                    conn.execute("UPDATE reports SET triage_error='Excluded from this smoke test' WHERE id=1")

            ui.date_input(key="report-date-from").set_value(date.today() + timedelta(days=1)).run(timeout=30)
            assert not ui.radio and not ui.exception
            ui.date_input(key="report-date-from").set_value(None).run(timeout=30)
            ui.radio(key="message-list").set_value(2).run(timeout=30)
            assert [event["event"] for event in store.activity(2)] == [
                "review_complete", "proposal_updated", "proposal_ready", "triage_started",
            ]
            click(ui, "Delete report")
            assert any(report["id"] == 2 for report in store.reports())
            ui.button(key="confirm-delete").click().run(timeout=30)
            assert not ui.exception and all(report["id"] != 2 for report in store.reports())
            assert any(job["created_from"] == 2 for job in store.jobs())

            ui.button_group(key="workspace").set_value("Jobs").run(timeout=30)
            edit_job_status(ui, 103, "Assigned")
            assert any(job["id"] == 103 and job["status"] == "assigned"
                       for job in store.jobs("Juniper Court / Flat 16 landing"))
            edit_job_status(ui, 103, "Handled")
            assert all(job["id"] != 103 for job in store.jobs("Juniper Court / Flat 16 landing"))
