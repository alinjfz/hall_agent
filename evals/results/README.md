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

The original check used the first eight rows in the [label fixture](../judge_labels.json). It used four complaint replies for development and four replies to a separate radiator repair question for testing. The complaint set starts with a Gemini reply from the Vertex AI smoke run. The other replies are synthetic. Each scenario includes a supported answer, an invented policy, an unsupported guarantee and a reply that leaves the question unanswered.

The expected labels were assigned with Codex before either judge run. For these labels, answering the resident measures whether the reply addresses the question. Grounding and unsupported promises score the accuracy of those answers separately. Independent human review and a larger sample of real drafts are still pending.

`gpt-6-luna` matched all 12 labels in [development](judge-contrast-dev.json) and all 12 in [testing](judge-contrast-test.json) with the same rubrics.

| Criterion | Failing test replies caught | Passing test replies accepted |
| --- | ---: | ---: |
| Grounding | 2/2 | 2/2 |
| Unsupported promises | 1/1 | 3/3 |
| Answers the resident | 1/1 | 3/3 |

### Rubric v2

Grounding now checks factual claims about both policy and the specific job. Covers the question requires every material part to be answered or identified as needing confirmation. The JSON still uses `answers_resident` for coverage so existing consumers can read it.

Six new synthetic replies check partial answers, invented job updates and explicit uncertainty. With these additions, GPT agreed with 20/21 expected verdicts in [development](judge-contrast-v2-dev.json) and 19/21 in [testing](judge-contrast-v2-test.json).

| Criterion | Failing test replies caught | Passing test replies accepted |
| --- | ---: | ---: |
| Grounding | 3/3 | 3/4 |
| Unsupported promises | 1/1 | 5/6 |
| Covers the question | 2/2 | 5/5 |

GPT treated “staff still need to confirm” as an unsupported claim in both splits, even though the reply used it to explain what the job record leaves open. On the test split it also treated an invented booking as an unsupported promise. The preset labels count the booking as an unsupported factual claim because the reply asserts a booking already happened.

The [full Vertex triage run](vertex-gpt-triage-v2.json) completed 195 trials across 65 cases. Deterministic checks passed 190 trials. The portable-light case proposed a repair instead of asking about working fixed lighting in all three trials. Two other cases had a route mismatch in one trial each. GPT scored six model drafts across two cases and passed all three criteria on all six drafts.

### Rubric v3 and routing fixes

Grounding now accepts a statement that staff need to confirm a fact left open by the evidence. The promise check treats a claim about an existing booking as factual and looks for a separate commitment about future attendance or completion. The judge matched all 21 expected verdicts on both the [development rerun](judge-contrast-v3-dev.json) and the [test rerun](judge-contrast-v3-test.json). These reused examples verify the fixes. A fresh holdout would measure how the revised rubrics generalize.

The [targeted Vertex rerun](vertex-routing-fixes.json) passed all 24 trials across the three failed cases and five controls. The policy now asks for clarification when portable lighting is the only evidence, sends explicit follow-ups without a verified matching repair to staff, rejects approval instructions as split issues and sends furniture misclassified as appliances to staff. It also restores articles and punctuation when the quoted split issue otherwise matches a complete sentence.

The [full post-fix run](vertex-full-post-fix.json) repeated all 65 cases × 3 trials on the same dataset fingerprint and model and passed 195/195 deterministic checks, with no missed hazards or false alarms. GPT scored six model drafts and passed 17 of 18 verdicts. It failed grounding on one complaint reply that told the resident to give "your job number" where the handbook says "any job number"; the deterministic checks passed that draft. The run used four workers behind the shared rate limiter, so its 7.3 s median latency includes time queued for a request slot and is not comparable with the earlier run's 3.6 s. Retrieval metrics now count only trials where a handbook search ran. Rescored that way, the earlier full run has recall@3 0.95 and hit@1 0.68 (previously reported as 0.61 and 0.43 because urgent handoffs, which skip the search, were counted as misses); the post-fix run matches.

```sh
for split in dev test; do
  LANGSMITH_TRACING=false .venv/bin/python -m evals.calibrate evals/judge_labels.json --split "$split"
done
```
