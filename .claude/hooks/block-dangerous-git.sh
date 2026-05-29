#!/bin/bash
set -euo pipefail

INPUT=$(cat)

if command -v jq >/dev/null 2>&1; then
  COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
else
  COMMAND=$(echo "$INPUT" | grep -o '"command":"[^"]*"' | sed 's/"command":"//;s/"$//')
fi

[ -z "${COMMAND:-}" ] && exit 0

DANGEROUS_PATTERNS=(
  "git push --force"
  "git push -f"
  "git reset --hard"
  "git clean -fd"
  "git clean -f"
  "git branch -D"
  "git checkout ."
  "git restore ."
  "--no-verify"
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
  if echo "$COMMAND" | grep -qF "$pattern" 2>/dev/null; then
    echo "BLOCKED: '$COMMAND' matches dangerous pattern '$pattern'. Ask the user before proceeding." >&2
    exit 2
  fi
done

exit 0
