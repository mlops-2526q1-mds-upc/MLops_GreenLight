#!/usr/bin/env bash
set -euo pipefail

# Runs Black (check), Ruff (lint), and Pylint on the mlops_greenlight package
# Produces per-tool logs and a concise summary at the end.

PROJECT_DIR="."
SRC_DIR="${PROJECT_DIR}/mlops_greenlight"
OUT_DIR="${PROJECT_DIR}/.lint_reports"
mkdir -p "${OUT_DIR}"

echo "Linting source at: ${SRC_DIR}"

# Black: check only (no write)
echo "Running Black (check)..."
BLACK_LOG="${OUT_DIR}/black.txt"
if black --check --diff "${SRC_DIR}" >"${BLACK_LOG}" 2>&1; then
  BLACK_STATUS=0
else
  BLACK_STATUS=$?
fi

# Ruff: lint with statistics
echo "Running Ruff..."
RUFF_LOG="${OUT_DIR}/ruff.txt"
if ruff check --statistics "${SRC_DIR}" >"${RUFF_LOG}" 2>&1; then
  RUFF_STATUS=0
else
  RUFF_STATUS=$?
fi

# Pylint
echo "Running Pylint..."
PYLINT_LOG="${OUT_DIR}/pylint.txt"
if pylint -j 0 "${SRC_DIR}" >"${PYLINT_LOG}" 2>&1; then
  PYLINT_STATUS=0
else
  PYLINT_STATUS=$?
fi

echo
echo "================ Summary ================"

# Black summary: number of files that would be reformatted
if [[ ${BLACK_STATUS} -eq 0 ]]; then
  BLACK_NEEDS=0
else
  BLACK_NEEDS=$(grep -E "would reformat " -c "${BLACK_LOG}" || true)
fi
if [[ -z "${BLACK_NEEDS}" ]]; then BLACK_NEEDS=0; fi
echo "Black: $( [[ ${BLACK_STATUS} -eq 0 ]] && echo PASS || echo FAIL ) — files needing format: ${BLACK_NEEDS}"

# Ruff summary: total issues and unique rules (from --statistics)
RUFF_RULE_LINES=$(grep -E "^[A-Z][0-9]{3}[[:space:]]+[0-9]+$" "${RUFF_LOG}" || true)
if [[ -n "${RUFF_RULE_LINES}" ]]; then
  RUFF_TOTAL=$(echo "${RUFF_RULE_LINES}" | awk '{s+=$2} END {print s+0}')
  RUFF_UNIQUE=$(echo "${RUFF_RULE_LINES}" | wc -l | awk '{print $1}')
else
  # Fallback: count raw issue lines
  RUFF_TOTAL=$(grep -E "^[^:]+:[0-9]+:[0-9]+: [A-Z][0-9]{3} " -c "${RUFF_LOG}" || true)
  RUFF_UNIQUE=$(grep -Eo "^[^:]+:[0-9]+:[0-9]+: [A-Z][0-9]{3} " "${RUFF_LOG}" | awk '{print $4}' | sort -u | wc -l | awk '{print $1}' || true)
fi
if [[ -z "${RUFF_TOTAL}" ]]; then RUFF_TOTAL=0; fi
if [[ -z "${RUFF_UNIQUE}" ]]; then RUFF_UNIQUE=0; fi
echo "Ruff: $( [[ ${RUFF_STATUS} -eq 0 ]] && echo PASS || echo FAIL ) — issues: ${RUFF_TOTAL}, rules: ${RUFF_UNIQUE}"

# Pylint summary: total messages and overall score
PYLINT_MSGS=$(grep -E "^[^:]+:[0-9]+:[0-9]+: [A-Z][0-9]{4}:" -c "${PYLINT_LOG}" || true)
if [[ -z "${PYLINT_MSGS}" ]]; then PYLINT_MSGS=0; fi
PYLINT_SCORE=$(grep -Eo "rated at [0-9]+\.[0-9]+/10|rated at [0-9]+/10" "${PYLINT_LOG}" | tail -n1 | awk '{print $3}' | tr -d '/10' || true)
if [[ -z "${PYLINT_SCORE}" ]]; then PYLINT_SCORE="n/a"; fi
echo "Pylint: $( [[ ${PYLINT_STATUS} -eq 0 ]] && echo PASS || echo FAIL ) — messages: ${PYLINT_MSGS}, score: ${PYLINT_SCORE}"

echo "Logs: ${OUT_DIR}/black.txt, ${OUT_DIR}/ruff.txt, ${OUT_DIR}/pylint.txt"

# Exit non-zero if any tool failed
if [[ ${BLACK_STATUS} -ne 0 || ${RUFF_STATUS} -ne 0 || ${PYLINT_STATUS} -ne 0 ]]; then
  exit 1
fi

exit 0


