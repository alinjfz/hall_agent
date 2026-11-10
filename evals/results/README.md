# Evaluation artifacts

The historical artifacts predate the v2 dataset and were checked in on 2026-11-09. Their exact run dates and revisions are not always recorded.

| File | Run | Revision and model | Coverage |
| --- | --- | --- | --- |
| [matching-baseline.json](matching-baseline.json) | Original matching baseline | Historical scripted agent; no live model | Complete eight-case scripted run |
| [chase-full-before.json](chase-full-before.json) | Earlier eight-case local run, old harness, full mode before changes | Historical; Qwen 3.5 9B | Complete for that chase suite |
| [chase-full-after.json](chase-full-after.json) | Earlier eight-case local run, old harness, full mode after changes | Historical; Qwen 3.5 9B | Complete for that chase suite |
| [chase-single-after.json](chase-single-after.json) | Earlier eight-case local run, old harness, single mode | Historical; Qwen 3.5 9B | Complete for that chase suite |
| [chase-specialist-after.json](chase-specialist-after.json) | Earlier eight-case local run, old harness, specialist mode | Historical; Qwen 3.5 9B | Complete for that chase suite |
| [baseline-smoke-overdue.json](baseline-smoke-overdue.json) | Original overdue chase | `3668279`; Qwen 3.5 9B | Smoke, three trials |
| [current-smoke-overdue.json](current-smoke-overdue.json) | Unified graph overdue chase | `534a22a` source era; Qwen 3.5 9B | Smoke, three trials |
| [curated-with-guidance.json](curated-with-guidance.json) | Contained shower drip with guidance | `534a22a` source era; Qwen 3.5 9B | Smoke, three trials |
| [curated-without-guidance.json](curated-without-guidance.json) | Same case without guidance | `534a22a` source era; Qwen 3.5 9B | Smoke, three trials |
| [chase-v2-without-followup.json](chase-v2-without-followup.json) | Lookup follow-up disabled | Polish working tree on `534a22a`; Qwen 3.5 9B, 2026-11-09 | Eight selected cases × three trials |
| [chase-v2-with-followup.json](chase-v2-with-followup.json) | Same chase slice with lookup follow-up enabled | Polish working tree on `534a22a`; Qwen 3.5 9B, 2026-11-09 | Eight selected cases × three trials |
| [baseline-hosted-chase.json](baseline-hosted-chase.json) | Hosted chase comparison, pinned original agent | `3668279`; Qwen 3.5 9B, 2026-11-09 | Three selected cases × three trials |
| [current-hosted-chase.json](current-hosted-chase.json) | Hosted chase comparison, current graph | Polish working tree on `534a22a`; Qwen 3.5 9B, 2026-11-09 | Three selected cases × three trials |
| [current-hosted-curated.json](current-hosted-curated.json) | Hosted curated smoke with guidance | Polish working tree on `534a22a`; Qwen 3.5 9B, 2026-11-09 | One case × three trials |

The two chase ablation arms share a fingerprint. Two handbook section labels were corrected afterwards for the full v2 run; the report and job inputs were unchanged.

## GPT judge contrast check

The [label fixture](../judge_labels.json) has four complaint replies for development and four replies to a separate radiator repair question for testing. The complaint set starts with a Gemini reply from the Vertex AI smoke run. The other replies are synthetic. Each scenario includes a supported answer, an invented policy, an unsupported guarantee and a reply that leaves the question unanswered.

The expected labels were assigned with Codex before either judge run. For these labels, answering the resident measures whether the reply addresses the question. Grounding and unsupported promises score the accuracy of those answers separately. Independent human review and a larger sample of real drafts are still pending.

`gpt-6-luna` matched all 12 labels in [development](judge-contrast-dev.json) and all 12 in [testing](judge-contrast-test.json) with the same rubrics.

| Criterion | Failing test replies caught | Passing test replies accepted |
| --- | ---: | ---: |
| Grounding | 2/2 | 2/2 |
| Unsupported promises | 1/1 | 3/3 |
| Answers the resident | 1/1 | 3/3 |

```sh
for split in dev test; do
  LANGSMITH_TRACING=false .venv/bin/python -m evals.calibrate evals/judge_labels.json --split "$split"
done
```
