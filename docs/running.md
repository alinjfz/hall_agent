# Running UniHelp

## Workspace and Studio

New databases start empty. For a sample inbox, set `HALL_SEED_SAMPLE=true` and use a separate, fresh `HALL_DB`. Set `HALL_REVIEWER_ID` to the workspace reviewer ID. To run the worker against the same database, set `HALL_BACKGROUND_WORKER=true` and use `.venv/bin/python -m halldesk.worker` (add `--once` to drain the queue and exit). `HALL_FALLBACK_MODEL` can name a second loaded LM Studio model for transient errors.

Run `make studio`, open the printed URL, and select **unihelp**. Start with `{"report_id":1}` or supply `location`, `subject`, and `message`. Resume the seeded escalation with `"approve"`, or use the action, version and values shown at the interrupt. Studio uses a separate SQLite workspace that resets when the server restarts. Stop the server with Ctrl+C.

## Live experiments

Use the same model and provider for the baseline and current agent. Run one experiment at a time on LM Studio. Set `LANGSMITH_API_KEY` to sync and run hosted experiments. The pinned baseline uses the same evaluation harness and fixture files.

```sh
make eval EVAL_ARGS="--langsmith hosted --sync-only"
./scripts/eval-baseline.sh --langsmith hosted --output "$PWD/evals/results/baseline-triage.json"
make eval EVAL_ARGS="--langsmith hosted --output evals/results/current-triage.json"
make eval-safety EVAL_ARGS="--langsmith hosted --output evals/results/current-safety.json"
```

Append `--case overdue` to run a single case after syncing the full suite. Repeat `--case` to select several. Local mode captures traces without uploading them. Use only synthetic or explicitly approved data for hosted evaluation: redaction covers some common contact details and names. Each hosted result records the experiment run ID and trace ID. Share selected synthetic runs individually if publishing evidence.

Vertex evals on the current graph allow 60 seconds for interpretation, 45 for safety and 45 for drafting. Override the budgets in that order with `--deadlines`, for example `make eval EVAL_ARGS="--agent vertex --case overdue --deadlines 90 60 60"`. Results record the selected budgets. The app uses its own shorter defaults.

The output records nomination counts, median latency and total model calls. Nomination is scored before Python verifies the IDs and does not gate route or safety checks.

## Handbook retrieval

Search is keyword-only until `HALL_EMBEDDING_MODEL` names an embedding model from the selected provider, for example `text-embedding-3-small` for OpenAI or an embedding model loaded in LM Studio. Each result records which half matched it, and the retriever run records `retrieval_mode`.

```sh
.venv/bin/python -m evals.retrieval_sweep
.venv/bin/python -m evals.retrieval_sweep --hybrid
```

The sweep needs no chat model. It reports recall@3, hit@1 and MRR for the report text alone and for the production query with the expected intent and category added. The first command sweeps the BM25 floor. The second needs the embedding service and sweeps the similarity floor; copy the chosen value to `MIN_SIMILARITY` in `halldesk/retrieval.py`.

Interrupted experiments leave a `.partial.jsonl` file. Only complete runs are eligible for the regression gate. Supply `--against` with a reviewed run or save it as `evals/results/last-approved.json`. Comparisons require matching model, dataset fingerprint and repetition counts. Safety reports include both trial counts and case counts: a hazard counts as caught only if it reaches the expected urgency in every trial, and a look-alike counts as flagged if it triggers an alert in any trial.

## Guidance export demonstration

The [curated fixture](../evals/curated.json) demonstrates the export format and how to toggle guidance. It contains one synthetic case; it is not evidence that guidance helps. The guidance example quotes its report verbatim, but both arms failed the verbatim-evidence check on the separate shower report.

Set `HALL_ENABLE_QUALITY_CHECKS=true` to expose the quality queue. Edits and rejections create draft cases with the report snapshot and reviewer change. Staff label the expected outcome; guidance also needs a corrected interpretation and a note. Similar reports cannot be approved in opposing sets. Guidance is retired when its recorded rule revision changes.

```sh
.venv/bin/python -m evals.curation halldesk.sqlite3 /tmp/unihelp-curated.json
make eval EVAL_ARGS="--curated /tmp/unihelp-curated.json --output /tmp/with-guidance.json"
make eval EVAL_ARGS="--curated /tmp/unihelp-curated.json --without-guidance --output /tmp/without-guidance.json"
```

## Judge calibration

For judge scores, set `HALL_JUDGE_MODEL=google/gemma-4-e4b` and pass `--judge`. Have a reviewer label drafts before looking at judge scores. The calibration input is a JSON list with `id`, `split` (`dev` or `test`), `report`, `reply`, `sections`, `job` and three Boolean `labels`: `grounded`, `no_unsupported_promise`, and `answers_resident`. Tune on dev, then run `.venv/bin/python -m evals.calibrate labels.json --split test`.
