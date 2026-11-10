#!/bin/sh
# Evaluate the original agent with the same live fixtures and scoring harness.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git -C "$root" archive 3668279af4bd1ac3b44226f55ac938cba7b30409 halldesk | tar -x -C "$scratch"
# Match the current harness's package and environment names in the scratch copy.
mv "$scratch/halldesk" "$scratch/unihelp"
find "$scratch/unihelp" -name '*.py' -exec perl -pi -e 's/\bhalldesk\b/unihelp/g; s/\bHALL_(?=[A-Z])/UNIHELP_/g' {} +
mkdir "$scratch/evals"
cp "$root"/evals/run.py "$root"/evals/evaluators.py "$root"/evals/dataset.json "$root"/evals/safety_holdout.json "$scratch/evals/"
# The reply and evidence verifiers belong to the shared harness; the pinned agent never imports them.
cp "$root/unihelp/replies.py" "$root/unihelp/evidence.py" "$scratch/unihelp/"
if [ -f "$root/.env" ]; then ln -s "$root/.env" "$scratch/.env"; fi
cd "$scratch"
"$root/.venv/bin/python" -m evals.run "$@"
