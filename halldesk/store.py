import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from unicodedata import category, normalize
from uuid import uuid4
from zoneinfo import ZoneInfo

from halldesk.context import REPAIR_CATEGORIES, message_key, same_policy, similar_messages

LONDON = ZoneInfo("Europe/London")


def utcnow():
    return datetime.now(timezone.utc)


def clean_report_text(value):
    value = normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    bidi_overrides = "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
    return "".join(
        char for char in value
        if char in "\n\t" or (category(char) != "Cc" and char not in bidi_overrides)
    ).strip()


def is_overdue(job, now):
    return job["status"] in {"open", "assigned"} and now > datetime.fromisoformat(job["deadline"])


RESIDENCE = "Juniper Court"
FLAT_NUMBER = re.compile(r"^flat\s+(\d+)\b", re.IGNORECASE)
RETIRED_FLAT = 111
HANDOFFS = ("escalate", "draft", "create", "job_question", "lighting_question", "plumbing_question", "access_question", "power_question", "split", "manual")
LABELS = ["communal_area", "ceiling_light", "wall_light"]


class ReviewError(ValueError):
    pass


class Store:
    def __init__(self, path, clock=utcnow):
        self.path = path
        self.clock = clock

    @contextmanager
    def transaction(self, write=False):
        db = sqlite3.connect(self.path, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def seed(self, samples=True):
        with self.transaction(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY, location TEXT NOT NULL, message TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 0, proposal TEXT, result TEXT, trace_id TEXT
            )""")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(reports)")}
            for column in ("subject", "received_at", "triage_claim", "claim_until", "triage_error", "deleted_at", "interpretation"):
                if column not in columns:
                    db.execute(f"ALTER TABLE reports ADD COLUMN {column} TEXT")
            if "triage_attempts" not in columns:
                db.execute("ALTER TABLE reports ADD COLUMN triage_attempts INTEGER NOT NULL DEFAULT 0")
            if "parent_report_id" not in columns:
                db.execute("ALTER TABLE reports ADD COLUMN parent_report_id INTEGER REFERENCES reports(id)")
            db.execute("""CREATE TABLE IF NOT EXISTS report_activity (
                id INTEGER PRIMARY KEY, report_id INTEGER NOT NULL REFERENCES reports(id),
                at TEXT NOT NULL, event TEXT NOT NULL, details TEXT NOT NULL
            )""")
            if "reviewer_id" not in {row["name"] for row in db.execute("PRAGMA table_info(report_activity)")}:
                db.execute("ALTER TABLE report_activity ADD COLUMN reviewer_id TEXT NOT NULL DEFAULT 'system'")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY, location TEXT NOT NULL, fitting TEXT NOT NULL,
                deadline TEXT NOT NULL, priority TEXT NOT NULL, status TEXT NOT NULL,
                category TEXT NOT NULL, labels TEXT NOT NULL, escalated_at TEXT,
                created_from INTEGER UNIQUE REFERENCES reports(id)
            )""")
            if "assignee" not in {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}:
                db.execute("ALTER TABLE jobs ADD COLUMN assignee TEXT")
            db.execute("""CREATE TABLE IF NOT EXISTS eval_cases (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL, message_key TEXT NOT NULL UNIQUE,
                snapshot TEXT NOT NULL, expected TEXT NOT NULL, notes TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('draft','holdout','guidance','retired'))
            )""")
            eval_columns = {row["name"] for row in db.execute("PRAGMA table_info(eval_cases)")}
            for column in ("interpretation", "policy_revisions", "trace_id"):
                if column not in eval_columns:
                    db.execute(f"ALTER TABLE eval_cases ADD COLUMN {column} TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS reports_location ON reports(location)")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_location_status ON jobs(location, status)")
            db.execute("CREATE INDEX IF NOT EXISTS report_activity_report ON report_activity(report_id)")
            if not samples or db.execute("SELECT 1 FROM reports LIMIT 1").fetchone():
                return
            db.execute("INSERT OR IGNORE INTO reports (id,location,message) VALUES (?,?,?)",
                       (1, f"{RESIDENCE} / Flat 12 kitchen",
                        "I'm chasing the ceiling light repair I already reported. Is there an update?"))
            db.execute("""INSERT OR IGNORE INTO jobs
                (id,location,fitting,deadline,priority,status,category,labels)
                VALUES (?,?,?,?,?,'open','lighting',?)""",
                       (101, f"{RESIDENCE} / Flat 12 kitchen", "ceiling light",
                        (self.clock() - timedelta(days=2)).isoformat(), "P3",
                        '["communal_area","ceiling_light"]'))

            db.executemany("INSERT OR IGNORE INTO reports (id,location,message) VALUES (?,?,?)", [
                (2, f"{RESIDENCE} / Flat 14 kitchen",
                 "The ceiling light has stopped working. The wall light in the kitchen still works."),
                (3, f"{RESIDENCE} / Flat 16 landing",
                 "Any update on the light repair I reported on the landing?"),
            ])
            db.executemany("""INSERT OR IGNORE INTO jobs
                (id,location,fitting,deadline,priority,status,category,labels)
                VALUES (?,?,?,?,?,'open','lighting',?)""", [
                (102, f"{RESIDENCE} / Flat 16 landing", "ceiling light",
                 (self.clock() - timedelta(days=1)).isoformat(), "P3",
                 '["communal_area","ceiling_light"]'),
                (103, f"{RESIDENCE} / Flat 16 landing", "wall light",
                 (self.clock() + timedelta(days=2)).isoformat(), "P3",
                 '["communal_area","wall_light"]'),
            ])

            demo_reports = [
                (4, "Flat 16 landing", "Is the wall light still scheduled?",
                 "Can you check when the wall light repair I reported is due?"),
                (5, "Flat 18 hallway", "The hallway is completely dark",
                 "The ceiling light has stopped working. No other lights work in the hallway."),
                (6, "Flat 21 study", "A light has gone out in the study",
                 "The desk light has stopped working."),
                (7, "Flat 23 kitchen", "Sparks from a kitchen light",
                 "The ceiling light still flickers and I saw sparks coming from the fitting."),
                (8, "Flat 12 kitchen", "Following up on the wall light",
                 "I reported the wall light last week. Could you check where that repair is up to?"),
            ]
            db.executemany("""INSERT OR IGNORE INTO reports (id,location,subject,message,received_at)
                VALUES (?,?,?,?,?)""", [
                (report_id, f"{RESIDENCE} / {location}", subject, message, self.clock().isoformat())
                for report_id, location, subject, message in demo_reports
            ])
            db.executemany("""UPDATE reports SET subject=COALESCE(subject, ?),
                received_at=COALESCE(received_at, ?) WHERE id=?""", [
                (subject, self.clock().isoformat(), report_id) for report_id, subject in (
                    (1, "Ceiling light repair chase-up"), (2, "Kitchen ceiling light has stopped working"),
                    (3, "Any update on the landing repair?"),
                )
            ])

    @staticmethod
    def read_locations(db):
        return db.execute("SELECT location FROM reports UNION SELECT location FROM jobs ORDER BY location COLLATE NOCASE")

    def locations(self):
        with self.transaction() as db:
            existing = [row["location"].removeprefix(f"{RESIDENCE} / ") for row in self.read_locations(db)]
        names = {f"flat {number}": f"Flat {number}" for number in range(1, 31)}
        for name in existing:
            match = FLAT_NUMBER.match(name)
            if not match or int(match[1]) != RETIRED_FLAT:
                names.setdefault(name.casefold(), name)

        def location_order(name):
            match = FLAT_NUMBER.match(name)
            return (match is None, int(match[1]) if match else 0, name.casefold())

        return sorted(names.values(), key=location_order)

    def add_report(self, location, subject, message):
        location = " ".join(clean_report_text(location or "").split())
        subject = " ".join(clean_report_text(subject).split())
        message = clean_report_text(message)
        if not (0 < len(location) <= 100 and 0 < len(subject) <= 120 and 0 < len(message) <= 4000):
            raise ReviewError("Enter a location, a subject, and a report within the displayed limits.")
        match = FLAT_NUMBER.match(location)
        if match and int(match[1]) == RETIRED_FLAT:
            raise ReviewError("Flat 111 is closed to new reports. Choose another location.")
        with self.transaction(write=True) as db:
            full_location = f"{RESIDENCE} / {location}"
            full_location = next((row["location"] for row in self.read_locations(db)
                                  if row["location"].casefold() == full_location.casefold()), full_location)
            now = self.clock()
            row = db.execute("""INSERT INTO reports (location,subject,message,received_at) VALUES (?,?,?,?)""",
                             (full_location, subject, message, now.isoformat()))
            report_id = row.lastrowid
            self.intake_safety(db, report_id, full_location, message, now)
            return report_id

    def intake_safety(self, db, report_id, location, message, now):
        from halldesk.safety import safety_signals

        if not safety_signals(message):
            return
        from halldesk.context import retrieve_policy
        from halldesk.policy import build_proposal

        context = {"policies": retrieve_policy(), "jobs": self.read_jobs(db, location), "examples": [],
                   "at": now.isoformat()}
        proposal = build_proposal({"message": message, "location": location}, context, None)
        db.execute("UPDATE reports SET proposal=?, version=1 WHERE id=?", (json.dumps(proposal), report_id))
        self.record_activity(db, report_id, "proposal_ready", {"kind": proposal["kind"], "urgency": proposal["urgency"]})
        self.record_emergency_reply(db, report_id, proposal)

    def record_emergency_reply(self, db, report_id, proposal):
        if (proposal.get("urgency") == "emergency" and proposal.get("reply")
                and not db.execute("SELECT 1 FROM report_activity WHERE report_id=? AND event='emergency_reply'",
                                   (report_id,)).fetchone()):
            self.record_activity(db, report_id, "emergency_reply",
                                 {"author": "staff", "body": proposal["reply"], "automatic": True})

    def record_triage_attempt(self, report_id, version):
        with self.transaction(write=True) as db:
            db.execute("UPDATE reports SET triage_attempts=triage_attempts+1 WHERE id=? AND version=?",
                       (report_id, version))


    def save_eval_draft(self, report, expected, notes, *, interpretation=None):
        kind = expected.get("kind")
        if kind not in HANDOFFS:
            raise ReviewError("Choose an expected handoff.")
        with self.transaction(write=True) as db:
            if not db.execute("SELECT 1 FROM reports WHERE id=? AND deleted_at IS NULL",
                              (report["id"],)).fetchone():
                raise ReviewError("This report was deleted.")
            snapshot = (report.get("proposal") or {}).get("context_snapshot") or {
                "location": report["location"], "message": report["message"],
                "jobs": self.read_jobs(db, report["location"]), "at": self.clock().isoformat(),
            }
            jobs = snapshot["jobs"]
            if kind in {"escalate", "draft"} and expected.get("job") not in {job["id"] for job in jobs}:
                raise ReviewError("Choose an open job at this location.")
            if kind == "job_question" and (len(expected.get("candidates", [])) < 2 or
                    not set(expected["candidates"]).issubset({job["id"] for job in jobs})):
                raise ReviewError("Select at least two candidate jobs.")
            if kind == "create" and (expected.get("priority") not in {"P2", "P3"}
                    or type(expected.get("days")) is not int or not 1 <= expected["days"] <= 90
                    or expected.get("category", "lighting") not in REPAIR_CATEGORIES):
                raise ReviewError("Choose the expected repair category, priority, and target.")
            from halldesk.context import retrieve_policy
            from halldesk.interpretation import Interpretation

            if interpretation is not None:
                try:
                    interpretation = Interpretation.model_validate(interpretation).model_dump()
                    quotes = [interpretation.get(key) for key in ("evidence", "safety_evidence", "lighting_evidence")]
                    quotes += interpretation.get("issues") or []
                    if any(quote and quote not in snapshot["message"] for quote in quotes):
                        raise ValueError("Corrected evidence must occur in the report.")
                except ValueError as exc:
                    raise ReviewError("Enter a valid corrected interpretation.") from exc
            topic = (expected.get("category", "lighting") if kind == "create" else
                     {"escalate": "escalation", "draft": "follow_up", "job_question": "escalation",
                      "lighting_question": "lighting", "plumbing_question": "plumbing",
                      "access_question": "access", "power_question": "urgent", "split": "multiple"}.get(kind, "review"))
            policy = retrieve_policy().get(topic) or (report.get("proposal") or {}).get("rule")
            revisions = {policy["id"]: policy["revision"]} if policy else {}
            saved = db.execute("""INSERT INTO eval_cases
                (name,message_key,snapshot,expected,notes,status,interpretation,policy_revisions,trace_id)
                VALUES (?,?,?,?,?,'draft',?,?,?) ON CONFLICT(message_key) DO UPDATE SET
                expected=excluded.expected, notes=excluded.notes, interpretation=excluded.interpretation,
                policy_revisions=excluded.policy_revisions WHERE eval_cases.status='draft'""",
                               (report["subject"] or "Resident report", message_key(report["message"]),
                                json.dumps(snapshot), json.dumps(expected), notes.strip(),
                                json.dumps(interpretation) if interpretation else None,
                                json.dumps(revisions), report.get("trace_id")))
            if saved.rowcount != 1:
                raise ReviewError("An approved or retired case already exists for this report text.")

    def eval_cases(self, status=None):
        with self.transaction() as db:
            if status:
                rows = db.execute("SELECT * FROM eval_cases WHERE status=? ORDER BY id", (status,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM eval_cases ORDER BY id").fetchall()
        return [dict(row) | {key: json.loads(row[key]) if row[key] else None
                            for key in ("snapshot", "expected", "interpretation", "policy_revisions")}
                for row in rows]

    def active_guidance(self, policies):
        if not policies:
            return []
        revisions = {rule["id"]: rule["revision"] for rule in policies.values()}
        cases = self.eval_cases("guidance")
        stale = [case["id"] for case in cases if any(
            revisions.get(key) != version for key, version in (case["policy_revisions"] or {}).items())]
        if stale:
            with self.transaction(write=True) as db:
                db.executemany("UPDATE eval_cases SET status='retired' WHERE id=? AND status='guidance'",
                               [(case_id,) for case_id in stale])
        return [case for case in cases if case["id"] not in stale and case["interpretation"]]

    def decide_eval(self, case_id, status):
        if status not in {"holdout", "guidance", "retired"}:
            raise ReviewError("Choose how to use this case.")
        with self.transaction(write=True) as db:
            case = db.execute("SELECT * FROM eval_cases WHERE id=?", (case_id,)).fetchone()
            if not case:
                raise ReviewError("This case no longer exists.")
            if status in {"holdout", "guidance"}:
                if not json.loads(case["expected"]).get("kind"):
                    raise ReviewError("Label the expected handoff before approving this case.")
                if status == "guidance" and not (case["interpretation"] and case["notes"].strip()):
                    raise ReviewError("Add a corrected interpretation and reviewer note before approving guidance.")
                other = "holdout" if status == "guidance" else "guidance"
                messages = [json.loads(row["snapshot"])["message"] for row in db.execute(
                    "SELECT snapshot FROM eval_cases WHERE status=?", (other,))]
                if similar_messages(json.loads(case["snapshot"])["message"], messages):
                    raise ReviewError(f"A similar report is already approved as {other}.")
            updated = db.execute("""UPDATE eval_cases SET status=? WHERE id=? AND
                (status='draft' OR (?='retired' AND status IN ('holdout','guidance')))""",
                                 (status, case_id, status))
            if updated.rowcount != 1:
                raise ReviewError("This case has already been reviewed.")

    def claim_triage(self):
        now = self.clock()
        claim = str(uuid4())
        with self.transaction(write=True) as db:
            row = db.execute("""UPDATE reports SET triage_claim=?, claim_until=? WHERE id=(
                SELECT id FROM reports WHERE proposal IS NULL AND result IS NULL AND triage_error IS NULL
                AND deleted_at IS NULL AND (triage_claim IS NULL OR claim_until <= ?) ORDER BY id LIMIT 1
            ) RETURNING id""", (claim, (now + timedelta(minutes=20)).isoformat(), now.isoformat())).fetchone()
            if row:
                self.record_activity(db, row["id"], "triage_started", at=now.isoformat())
        return (row["id"], claim) if row else None

    def fail_triage(self, report_id, claim):
        with self.transaction(write=True) as db:
            updated = db.execute("""UPDATE reports SET triage_claim=NULL, claim_until=NULL, triage_error=?
                WHERE id=? AND triage_claim=?""", ("Automatic triage failed. Check the model setup and retry.",
                                                   report_id, claim))
            if updated.rowcount:
                self.record_activity(db, report_id, "triage_failed")

    def retry_triage(self, report_id=None):
        sql = """UPDATE reports SET triage_error=NULL
            WHERE triage_error IS NOT NULL AND proposal IS NULL AND result IS NULL AND deleted_at IS NULL"""
        params = ()
        if report_id is not None:
            sql += " AND id=?"
            params = (report_id,)
        with self.transaction(write=True) as db:
            requeued = db.execute(sql + " RETURNING id", params).fetchall()
            for row in requeued:
                self.record_activity(db, row["id"], "requeued")
            return len(requeued)

    def requeue_review(self, report_id, version):
        with self.transaction(write=True) as db:
            updated = db.execute("""UPDATE reports SET proposal=NULL, version=version+1,
                trace_id=NULL, triage_error=NULL
                WHERE id=? AND version=? AND result IS NULL AND deleted_at IS NULL""",
                                 (report_id, version))
            if updated.rowcount != 1:
                raise ReviewError("Review changed. Reload the report.")
            self.record_activity(db, report_id, "requeued")

    def reset_report_queue(self):
        with self.transaction(write=True) as db:
            legacy_receipts = db.execute("""SELECT reports.id, reports.result FROM reports
                WHERE deleted_at IS NULL AND result IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM report_activity
                    WHERE report_id=reports.id AND event='review_complete'
                )""").fetchall()
            for row in legacy_receipts:
                result = json.loads(row["result"])
                result.setdefault("at", None)
                self.record_activity(db, row["id"], "review_complete", result, at=result.get("at"))
            requeued = db.execute("""UPDATE reports SET proposal=NULL, result=NULL, version=version+1,
                trace_id=NULL, triage_error=NULL, triage_claim=NULL, claim_until=NULL
                WHERE deleted_at IS NULL RETURNING id""").fetchall()
            for row in requeued:
                self.record_activity(db, row["id"], "requeued")
            return len(requeued)

    def delete_report(self, report_id, version):
        with self.transaction(write=True) as db:
            updated = db.execute("""UPDATE reports SET deleted_at=?, version=version+1,
                triage_claim=NULL, claim_until=NULL WHERE id=? AND version=? AND deleted_at IS NULL""",
                                 (self.clock().isoformat(), report_id, version))
            if updated.rowcount != 1:
                raise ReviewError("Review changed. Reload the report.")

    @staticmethod
    def staff_id(reviewer_id=None):
        return reviewer_id or os.getenv("HALL_REVIEWER_ID") or "staff"

    def record_activity(self, db, report_id, event, details=None, at=None, *, reviewer_id="system"):
        db.execute("INSERT INTO report_activity (report_id,at,event,details,reviewer_id) VALUES (?,?,?,?,?)",
                   (report_id, at or self.clock().isoformat(), event, json.dumps(details or {}), reviewer_id))

    def activity(self, report_id):
        with self.transaction() as db:
            rows = db.execute("SELECT * FROM report_activity WHERE report_id=? ORDER BY id DESC", (report_id,))
            return [dict(row) | {"details": json.loads(row["details"])} for row in rows]

    def add_ticket_message(self, report_id, author, body, *, draft_version=None, decline=False, reviewer_id=None):
        body = clean_report_text(body)
        if (author not in {"resident", "staff"} or (decline and draft_version is None)
                or (not decline and not 0 < len(body) <= 4000)):
            raise ReviewError("Enter a message of up to 4,000 characters.")
        with self.transaction(write=True) as db:
            report = db.execute("SELECT * FROM reports WHERE id=? AND deleted_at IS NULL", (report_id,)).fetchone()
            if report is None:
                raise ReviewError("This report is no longer available.")
            event = "conversation_message"
            if draft_version is not None:
                proposal = json.loads(report["proposal"] or "{}")
                result = json.loads(report["result"] or "{}")
                if (author != "staff" or report["version"] != draft_version or result.get("decision") == "reject"
                        or not (proposal.get("reply") or proposal.get("question"))):
                    raise ReviewError("This reply draft changed. Reload the report.")
                reviewed = db.execute("""SELECT 1 FROM report_activity WHERE report_id=?
                    AND event IN ('reply_approved', 'reply_declined')
                    AND json_extract(details, '$.version')=?""", (report_id, draft_version)).fetchone()
                if reviewed:
                    raise ReviewError("This draft has already been reviewed.")
                event = "reply_declined" if decline else "reply_approved"
                original_draft = proposal.get("reply") or proposal.get("question")
                if decline or body != original_draft:
                    revised = None if decline else proposal | {"reviewer_edits": {"reply": body}}
                    self.record_eval_draft(db, report, "Reviewer declined or edited the resident reply.", revised)
            self.record_activity(db, report_id, event, {
                "author": author, "body": "" if decline else body, "version": draft_version,
            }, reviewer_id=self.staff_id(reviewer_id) if author == "staff" else "resident")

    @staticmethod
    def decode_report(row):
        report = dict(row)
        for key in ("proposal", "result", "interpretation"):
            report[key] = json.loads(report[key]) if report[key] else None
        return report

    def reports(self):
        with self.transaction() as db:
            rows = db.execute("SELECT * FROM reports WHERE deleted_at IS NULL ORDER BY id").fetchall()
        return [self.decode_report(row) for row in rows]

    def report(self, report_id):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM reports WHERE id=? AND deleted_at IS NULL", (report_id,)).fetchone()
        if row is None:
            raise ReviewError("This report is no longer available.")
        return self.decode_report(row)

    def previous_matching_report(self, report):
        key = message_key(report["message"])
        with self.transaction() as db:
            earlier = db.execute(
                "SELECT id, message FROM reports WHERE location=? AND id<? AND deleted_at IS NULL ORDER BY id DESC",
                (report["location"], report["id"]),
            )
            return next((row["id"] for row in earlier if message_key(row["message"]) == key), None)

    @staticmethod
    def read_jobs(db, location=None):
        sql = "SELECT * FROM jobs"
        rows = (db.execute(sql + " WHERE location=? AND status IN ('open', 'assigned') ORDER BY id",
                           (location,)) if location is not None else db.execute(sql + " ORDER BY id"))
        return [
            {key: row[key] for key in row.keys() if key != "assignee" or row[key] is not None}
            | {"labels": json.loads(row["labels"])}
            for row in rows
        ]

    def jobs(self, location=None):
        with self.transaction() as db:
            return self.read_jobs(db, location)

    def update_job_status(self, job, status):
        if status not in {"open", "assigned", "handled", "declined"}:
            raise ReviewError("Choose a valid job status.")
        with self.transaction(write=True) as db:
            updated = db.execute("UPDATE jobs SET status=? WHERE id=? AND status=?",
                                 (status, job["id"], job["status"]))
            if updated.rowcount != 1:
                raise ReviewError("This job was updated. Reload it before making another change.")

    def open_job_reports(self, location):
        with self.transaction() as db:
            rows = db.execute("""SELECT jobs.id AS job_id, reports.id AS report_id, reports.message
                FROM jobs JOIN reports ON reports.id=jobs.created_from
                WHERE jobs.location=? AND reports.location=jobs.location AND jobs.status IN ('open', 'assigned')
                ORDER BY jobs.id LIMIT 25""", (location,))
            return {row["job_id"]: {"report_id": row["report_id"], "message": row["message"][:500]}
                    for row in rows}

    def save_proposal(self, report, proposal, trace_id=None, *, reviewer_id=None, interpretation=None):
        with self.transaction(write=True) as db:
            updated = db.execute("""UPDATE reports SET proposal=?, version=version+1, trace_id=COALESCE(?,trace_id),
                triage_claim=NULL, claim_until=NULL, triage_error=NULL, interpretation=COALESCE(?,interpretation)
                WHERE id=? AND version=? AND result IS NULL AND triage_claim IS ? AND deleted_at IS NULL""",
                                 (json.dumps(proposal), trace_id, json.dumps(interpretation) if interpretation else None,
                                  report["id"], report["version"], report["triage_claim"]))
            if updated.rowcount != 1:
                raise ReviewError("Review changed. Reload the report.")
            self.record_activity(db, report["id"], "proposal_updated" if report["proposal"] else "proposal_ready", {
                "kind": proposal["kind"],
                "job_id": proposal.get("job", {}).get("id"),
                "destination": proposal.get("rule", {}).get("destination"),
                "deadline": proposal.get("job", {}).get("deadline"),
                "note": proposal.get("question") or proposal.get("reason"),
                "days": proposal.get("days"),
                "reviewer_edits": proposal.get("reviewer_edits"),
            }, reviewer_id=self.staff_id(reviewer_id) if report["proposal"] else "system")
            self.record_emergency_reply(db, report["id"], proposal)
            if report["proposal"]:
                self.record_eval_draft(db, report, "Reviewer revised the proposal.", proposal)

    def record_eval_draft(self, db, report, note, revised=None):
        proposal = json.loads(report["proposal"]) if isinstance(report["proposal"], str) else report["proposal"]
        if not proposal or not (snapshot := proposal.get("context_snapshot")):
            return
        snapshot = snapshot | {"review_change": {"before": proposal["kind"],
                                                  "after": revised["kind"] if revised else "rejected",
                                                  "edits": (revised or {}).get("reviewer_edits")}}
        rule = proposal.get("rule")
        revisions = {rule["id"]: rule["revision"]} if rule else {}
        db.execute("""INSERT INTO eval_cases (name,message_key,snapshot,expected,notes,status,policy_revisions,trace_id)
            VALUES (?,?,?,?,?,'draft',?,?) ON CONFLICT(message_key) DO NOTHING""",
            (report["subject"] or "Resident report", message_key(report["message"]), json.dumps(snapshot),
             "{}", note, json.dumps(revisions), report["trace_id"]))

    @staticmethod
    def escalate_job(db, report, proposal, now):
        jobs = Store.read_jobs(db, report["location"])
        job = next((item for item in jobs if item["id"] == proposal["job"]["id"]), None)
        if job != proposal["job"] or not is_overdue(job, now) or job["escalated_at"]:
            raise ReviewError("Job eligibility changed. Triage the report again.")
        db.execute("UPDATE jobs SET escalated_at=? WHERE id=?", (now.isoformat(), job["id"]))
        return {"action": "escalated", "job_id": job["id"],
                "destination": proposal["rule"]["destination"], "at": now.isoformat()}

    @staticmethod
    def create_job(db, report, proposal, now):
        existing = db.execute("SELECT id FROM jobs WHERE created_from=?", (report["id"],)).fetchone()
        if existing:
            raise ReviewError(f"This report already created job #{existing['id']}. Review that job in Jobs.")
        jobs = Store.read_jobs(db, report["location"])
        if any(job["fitting"].casefold() == proposal["fitting"].casefold() for job in jobs):
            raise ReviewError("An open job now covers this fitting. Triage the report again.")
        edits = proposal.get("reviewer_edits", {})
        if due_date := edits.get("due_date"):
            deadline_at = datetime.combine(date.fromisoformat(due_date), time(23, 59, 59), LONDON)
            if deadline_at <= now:
                raise ReviewError("The edited deadline has passed. Save a new deadline before approving.")
        elif target := proposal.get("policy_deadline"):
            deadline_at = datetime.fromisoformat(target)
        else:
            deadline_at = now + timedelta(days=proposal["days"])
        deadline = deadline_at.isoformat()
        job = db.execute("""INSERT INTO jobs
            (location,fitting,deadline,priority,status,category,labels,created_from)
            VALUES (?,?,?,?,'open',?,?,?)""",
                         (report["location"], proposal["fitting"], deadline, proposal["priority"],
                          proposal.get("category", "lighting"), json.dumps(proposal["labels"]), report["id"]))
        result = {"action": "created_job", "job_id": job.lastrowid, "deadline": deadline}
        if edits:
            result["reviewer_edits"] = edits
        return result

    def finish(self, report_id, version, decision, load_rule, *, followup="", reviewer_id=None):
        with self.transaction(write=True) as db:
            report = db.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
            if not report or report["deleted_at"] or report["version"] != version:
                raise ReviewError("Review changed. Reload the report.")
            if report["result"]:
                result = json.loads(report["result"])
                if result["decision"] == decision:
                    return result
                raise ReviewError("This report is already handled.")
            if report["proposal"] is None:
                raise ReviewError("This report has no current proposal.")
            from halldesk.proposals import parse_proposal

            proposal_data = json.loads(report["proposal"])
            proposal = parse_proposal(proposal_data)
            now = self.clock()
            urgent_manual = proposal.kind == "manual" and bool(proposal.urgency)
            followups = list(dict.fromkeys(line.strip() for line in followup.splitlines() if line.strip()))
            if followups and (
                decision not in {"acknowledge", "technician_handoff"} or not urgent_manual
                or any(issue == report["message"].strip() or issue not in report["message"] for issue in followups)
            ):
                raise ReviewError("Copy the exact wording for each separate issue from the original report.")
            if decision == "reject":
                result = {"action": "rejected"}
                self.record_eval_draft(db, report, "Reviewer rejected the proposal.")
            elif decision == "approve" and proposal.kind in {"escalate", "create", "split"}:
                if not same_policy(load_rule(proposal.rule.topic), proposal.rule.model_dump(exclude_unset=True)):
                    raise ReviewError("Policy changed. Triage the report again.")
                action = {"escalate": self.escalate_job, "create": self.create_job,
                          "split": self.split_report}[proposal.kind]
                result = action(db, report, proposal_data, now)
            elif decision == "acknowledge" and proposal.kind in {"draft", "manual"}:
                result = {"action": proposal.kind}
            elif decision == "technician_handoff" and urgent_manual:
                result = {"action": "technician_handoff"}
            else:
                raise ReviewError("This proposal cannot be approved.")
            if followups:
                split = self.split_report(db, report, {"issues": followups}, now)
                result["report_ids"] = split["report_ids"]
            reviewer_id = self.staff_id(reviewer_id)
            result |= {"decision": decision, "version": version, "at": now.isoformat(), "reviewer_id": reviewer_id}
            db.execute("UPDATE reports SET result=? WHERE id=?", (json.dumps(result), report_id))
            self.record_activity(db, report_id, "review_complete", result, result["at"], reviewer_id=reviewer_id)
            return result

    def split_report(self, db, report, proposal, now):
        if db.execute("SELECT 1 FROM reports WHERE parent_report_id=?", (report["id"],)).fetchone():
            raise ReviewError("This report already has separate reports from a previous review.")
        children = []
        for index, issue in enumerate(proposal["issues"], start=1):
            child = db.execute("""INSERT INTO reports (location,subject,message,received_at,parent_report_id)
                VALUES (?,?,?,?,?)""", (report["location"], f"Report #{report['id']} · Issue {index}",
                                        issue, report["received_at"] or now.isoformat(), report["id"])).lastrowid
            self.intake_safety(db, child, report["location"], issue, now)
            children.append(child)
        return {"action": "split_report", "report_ids": children}
