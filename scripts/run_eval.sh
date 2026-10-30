#!/usr/bin/env bash
# Usage: run_eval.sh ENV METHOD TAG WORKERS SKILL [EVAL_SET ...]
# Resume existing episode artifacts; never erase successful results.
set -euo pipefail
if [[ ${1:-} == --help || $# -lt 5 ]]; then
  echo 'Usage: run_eval.sh ENV METHOD TAG WORKERS SKILL [EVAL_SET ...]'
  exit 0
fi
ENVN=$1; METHOD=$2; TAG=$3; WORKERS=$4; SKILL=$5; shift 5
EMB=${EMB:-/data/EM-Benchmark}
case "$ENVN" in
  eb-alf)
    source "${EMB}/env.sh"
    PYBIN=${EB_ALF_PYTHON:-${EMB}/venv_alf/bin/python}
    DEFAULT_SETS=(base common_sense complex_instruction spatial visual_appearance long_horizon)
    EXPECTED=50
    ;;
  eb-hab)
    source "${EMB}/env.sh"
    PYBIN=${EB_HAB_PYTHON:-${PY:?env.sh must define PY}}
    DEFAULT_SETS=(base common_sense complex_instruction spatial_relationship visual_appearance long_horizon)
    EXPECTED=50
    ;;
  eb-man)
    source "${EMB}/env_man.sh"
    PYBIN=${EB_MAN_PYTHON:-${PY:?env_man.sh must define PY}}
    DEFAULT_SETS=(base common_sense complex spatial visual)
    EXPECTED=0
    ;;
  *) echo "Unsupported domain: $ENVN" >&2; exit 2;;
esac
if [[ $# -eq 0 ]]; then set -- "${DEFAULT_SETS[@]}"; fi
[[ -x "$PYBIN" ]] || { echo "Missing Python: $PYBIN" >&2; exit 2; }
if [[ "$ENVN" == eb-alf ]]; then
  "$PYBIN" -c 'from importlib.metadata import version; assert version("werkzeug")=="0.16.1", "ALF requires its compatible venv"'
fi
EXTRA=()
if [[ "$METHOD" == method_a ]]; then
  [[ -s "$SKILL" ]] || { echo "Missing or empty skill: $SKILL" >&2; exit 2; }
  EXTRA=(--skill-file "$SKILL")
fi
mkdir -p "${EMB}/logs"
cd "${EMB}/em_skillopt"
for S in "$@"; do
  OUT="${EMB}/artifacts/${TAG}/${S}"
  LOG="${EMB}/logs/${TAG}_${S}_resume.log"
  printf '%s env=%s set=%s python=%s workers=%s\n' "$(date -Is)" "$ENVN" "$S" "$PYBIN" "$WORKERS" >> "$LOG"
  "$PYBIN" -u -m em_skillopt.cli.evaluate --env "$ENVN" --method "$METHOD" \
    --eval-set "$S" --split test --limit 0 --parallel-workers "$WORKERS" \
    --proxy "" --out "$OUT" "${EXTRA[@]}" >> "$LOG" 2>&1
  # Exit status alone does not establish completeness: continue_on_error can
  # omit infrastructure failures, and resume summaries may cover only new rows.
  "$PYBIN" - "$OUT" "$EXPECTED" <<'PY'
import json, pathlib, sys
root, expected = pathlib.Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(p.read_text()) for p in (root/'episodes').glob('*.json')]
ids = [r['spec']['episode_id'] for r in rows]
assert len(set(ids)) == len(ids), 'Duplicate episode IDs'
print(f'{root.name}: persisted={len(rows)}, successes={sum(bool(r["success"]) for r in rows)}')
if expected:
    indices = {r['spec']['index'] for r in rows}
    missing = sorted(set(range(expected)) - indices)
    assert not missing and len(rows) == expected, f'Incomplete test: missing {missing}'
else:
    assert rows, 'No persisted test episodes; verify domain-specific expected manifest separately'
PY
done
