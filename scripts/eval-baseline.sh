#!/bin/sh
# Evaluate the original agent with the same live fixtures and scoring harness.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git -C "$root" archive 3668279af4bd1ac3b44226f55ac938cba7b30409 halldesk | tar -x -C "$scratch"
mkdir "$scratch/evals"
cp "$root"/evals/run.py "$root"/evals/evaluators.py "$root"/evals/dataset.json "$root"/evals/safety_holdout.json "$scratch/evals/"
if [ -f "$root/.env" ]; then ln -s "$root/.env" "$scratch/.env"; fi
cd "$scratch"
HALL_PROVIDER=lmstudio "$root/.venv/bin/python" -m evals.run "$@"
