"""Build a standalone HTML preview of eval cases or a report from saved results."""

import argparse
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = {"triage": "dataset.json", "safety": "safety_holdout.json", "curated": "curated.json"}
SUITES = {"triage": "Triage", "safety": "Safety holdout", "curated": "Curated holdout"}


def label(key):
    return str(key).replace("_", " ").replace("-", " ").capitalize()


def display(value):
    if value is None:
        return "None"
    if isinstance(value, (bool, dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def facts(values):
    rows = [f"<div><dt>{escape(name)}</dt><dd>{escape(display(value))}</dd></div>"
            for name, value in values]
    return f'<dl class="facts">{"".join(rows)}</dl>'


def json_details(title, value):
    formatted = escape(json.dumps(value, indent=2, ensure_ascii=False))
    return f'<details class="detail"><summary>{escape(title)}</summary><pre>{formatted}</pre></details>'


def stat(title, value):
    return f'<div class="stat"><span class="meta">{escape(title)}</span><strong>{escape(str(value))}</strong></div>'


def panel(title, content):
    return f'<section class="panel"><h2>{escape(title)}</h2>{content}</section>'


def bar(name, count, maximum, caption):
    return (
        f'<div class="metric"><span>{escape(name)}</span>'
        f'<span class="track" aria-hidden="true"><i style="width:{100 * count / maximum:.2f}%"></i></span>'
        f'<span>{escape(caption)}</span></div>'
    )


def distribution(title, values):
    counts = Counter(values)
    return panel(title, "".join(
        bar(name, count, counts.total(), str(count)) for name, count in counts.most_common()
    ))


def explorer(scopes):
    options = "".join(f'<option value="{escape(key)}">{escape(name)}</option>' for key, name in scopes)
    return f"""
    <section class="explorer" aria-labelledby="case-heading">
    <h2 id="case-heading">Case explorer</h2>
    <div class="toolbar" hidden>
      <label>Search cases <input id="query" type="search" placeholder="Case ID, message, route or policy"></label>
      <label>Group <select id="scope"><option value="">All groups</option>{options}</select></label>
    </div>
    <p class="hint" id="visible-count" aria-live="polite"></p>
    """


def preview(cases, sources, guidance_count):
    groups = defaultdict(list)
    for suite, case in cases:
        groups[suite, case["slice"]].append(case)
    scope_names = {key: f"{SUITES[key[0]]} / {label(key[1])}" for key in groups}
    outputs = [case["outputs"] for _, case in cases]
    yield '<div class="stats">'
    yield stat("Fixture cases", len(cases))
    yield stat("Suites shown", len({suite for suite, _ in cases}))
    yield stat("Urgent or emergency cases", sum(row.get("urgency") in ("urgent", "emergency") for row in outputs))
    if "curated" in sources:
        yield stat("Curated guidance examples", guidance_count)
    yield "</div>"
    yield '<div class="overview">'
    yield distribution("Cases by suite and slice", (scope_names[suite, case["slice"]] for suite, case in cases))
    yield distribution("Expected routes", (
        label(row["route"]) if row.get("route") else "Route not asserted" for row in outputs
    ))
    yield distribution("Expected urgency", (
        label(row["urgency"]) if row.get("urgency") else "No urgency" if "urgency" in row else "Not asserted"
        for row in outputs
    ))
    yield "</div>"
    scopes = [(str(index), scope_names[key]) for index, key in enumerate(groups)]
    yield explorer(scopes)
    for (scope, name), (_, group) in zip(scopes, groups.items()):
        yield f'<section class="group" data-scope="{scope}"><h3>{escape(name)} · {len(group)}</h3>'
        for case in group:
            inputs = case["inputs"]
            expected = case["outputs"]
            tags = [label(expected["route"]) if expected.get("route") else "Route not asserted"]
            if expected.get("urgency"):
                tags.append(label(expected["urgency"]))
            badges = "".join(f'<span class="tag">{escape(tag)}</span>' for tag in tags)
            yield (
                f'<details class="case"><summary><strong>{escape(case["id"])}</strong>'
                f'<span class="tags">{badges}</span></summary><div class="case-body">'
                f'<h3>Resident message</h3><blockquote>{escape(inputs["message"])}</blockquote>'
            )
            yield facts([("Location", inputs.get("location")), ("Fixture clock", inputs.get("clock"))])
            yield "<h3>Expected outcome</h3>"
            yield facts((label(key), value) for key, value in expected.items())
            policy = {label(key): case[key] for key in ("policy_topics", "policy_sections", "policy_revisions") if case.get(key)}
            if policy:
                yield "<h3>Policy references</h3>"
                yield facts(policy.items())
            yield json_details("Full fixture inputs and open jobs", inputs)
            yield "</div></details>"
        yield "</section>"
    yield "</section>"


def trial_status(row):
    if row.get("error"):
        return "Error"
    checks = [value for value in row.get("checks", {}).values() if value is not None]
    if not checks:
        return "Unscored"
    return "Pass" if all(checks) else "Fail"


def scorecard(data):
    for key, title in (("safety", "Safety by trial"), ("safety_by_case", "Safety by case")):
        if data.get(key):
            for stage, metrics in data[key].items():
                yield panel(f"{title} · {label(stage)}", facts((label(k), v) for k, v in metrics.items()))
    retrieval = [(title, f"{data[key]:.0%}") for key, title in (
        ("retrieval_recall_at_3", "Recall at 3"), ("retrieval_hit_at_1", "Hit at 1")
    ) if isinstance(data.get(key), (int, float))]
    if retrieval:
        yield panel("Handbook retrieval", facts(retrieval))
    interpretation = data.get("interpretation")
    if interpretation:
        yield panel("Interpretation", facts((label(k), v) for k, v in interpretation.items()))


def report(data):
    rows = data["results"]
    counts = Counter(trial_status(row) for row in rows)
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row.get("slice") or "Unspecified slice"][row["case"]].append(row)
    yield '<div class="stats">'
    yield stat("Trials recorded", len(rows))
    yield stat("Passed", f'{counts["Pass"]} / {len(rows)}')
    yield stat("Failed checks", counts["Fail"])
    yield stat("Trial errors", counts["Error"])
    if counts["Unscored"]:
        yield stat("Unscored", counts["Unscored"])
    yield "</div>"
    yield '<div class="overview">'
    bars = []
    for name, cases in sorted(groups.items()):
        trials = [trial for case in cases.values() for trial in case]
        passed = sum(trial_status(row) == "Pass" for row in trials)
        bars.append(bar(label(name), passed, len(trials), f"{passed} / {len(trials)}"))
    yield panel("Passing trials by slice", "".join(bars))
    yield from scorecard(data)
    yield "</div>"
    scopes = [(str(index), label(name)) for index, name in enumerate(sorted(groups))]
    yield explorer(scopes)
    for (scope, name), (_, cases) in zip(scopes, sorted(groups.items())):
        yield f'<section class="group" data-scope="{scope}"><h3>{escape(name)} · {len(cases)}</h3>'
        for case, trials in sorted(cases.items()):
            passed = sum(trial_status(row) == "Pass" for row in trials)
            yield (
                f'<details class="case"><summary><strong>{escape(case)}</strong>'
                f'<span class="tags"><span class="tag">{passed} / {len(trials)} trials passed</span></span>'
                '</summary><div class="case-body">'
            )
            for index, row in enumerate(trials, 1):
                status = trial_status(row)
                style = {"Pass": "pass", "Fail": "fail", "Error": "fail"}.get(status, "meta")
                yield f'<details class="detail"><summary>Trial {index} · <strong class="{style}">{status}</strong></summary>'
                expected = row.get("expected")
                actual = row.get("actual")
                if isinstance(expected, dict):
                    comparable = actual if isinstance(actual, dict) else {}
                    yield "<h3>Expected and actual</h3>"
                    yield facts((label(key), f"{display(value)} → {display(comparable[key]) if key in comparable else 'Not recorded'}")
                                for key, value in expected.items())
                else:
                    yield facts([("Expected route", expected), ("Actual route", actual)])
                if isinstance(actual, dict) and actual.get("reply"):
                    yield f'<h3>Draft reply</h3><blockquote>{escape(actual["reply"])}</blockquote>'
                yield "<h3>Checks</h3>"
                yield facts((label(key), "Not applicable" if value is None else "Pass" if value else "Fail")
                            for key, value in row.get("checks", {}).items())
                for key, title in (("trajectory_checks", "Trajectory checks"), ("judge", "Judge results")):
                    if row.get(key):
                        yield json_details(title, row[key])
                yield json_details("Full trial record", row)
                yield "</details>"
            yield "</div></details>"
        yield "</section>"
    yield json_details("Run metadata", {key: value for key, value in data.items() if key != "results"})
    yield "</section>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--suite", choices=DATASETS, help="Preview one suite (default: all checked-in suites)")
    sources.add_argument("--curated", type=Path, help="Preview the cases in a custom curated export")
    sources.add_argument("--results", type=Path, help="Read a completed evals.run JSON report")
    parser.add_argument("--case", action="append", default=[], help="Preview only this case ID (repeatable)")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "eval-report.html")
    args = parser.parse_args()
    if args.results and args.case:
        parser.error("--case is only available for suite previews.")
    if args.output.suffix.lower() != ".html":
        parser.error("--output must be an .html file.")

    def load(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

    if args.results:
        data = load(args.results)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list) or not data["results"]:
            parser.error("Expected a completed run JSON file with trial results.")
        mode, title = "Saved run", "Evaluation results"
        model = data.get("model") or data.get("agent") or data.get("mode") or "Unspecified model"
        subtitle = " · ".join(filter(None, (data.get("suite"), data.get("provider"), model, args.results.name)))
        content = report(data)
    else:
        selected = (args.suite,) if args.suite else ("curated",) if args.curated else DATASETS
        paths = {suite: args.curated or HERE / DATASETS[suite] for suite in selected}
        cases = []
        guidance_count = 0
        for suite, path in paths.items():
            data = load(path)
            if suite == "curated" and isinstance(data, dict):
                guidance_count = len(data.get("guidance", []))
                data = data.get("cases")
            if not isinstance(data, list) or not data:
                parser.error(f"Expected a dataset with at least one case in {path}.")
            cases.extend((suite, case) for case in data)
        if unknown := set(args.case) - {case["id"] for _, case in cases}:
            parser.error(f"Unknown case IDs: {', '.join(sorted(unknown))}")
        cases = [(suite, case) for suite, case in cases if not args.case or case["id"] in args.case]
        mode, title = "Fixture preview", "Evaluation coverage"
        subtitle = "Expected outcomes from local fixtures. Live scores appear when you load a saved run."
        content = preview(cases, paths, guidance_count)

    header = (
        f'<header class="hero"><div class="eyebrow">{mode}</div><h1>{title}</h1>'
        f'<p class="meta">{escape(subtitle)}</p></header>'
    )
    template = (HERE / "visualize.html").read_text(encoding="utf-8")
    document = template.replace("<!-- REPORT -->", "\n".join([header, *content]), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
