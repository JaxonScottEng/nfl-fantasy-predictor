#!/usr/bin/env python
"""
PostToolUse hook: after an edit to a .py file under src/, re-run the
evaluation harness and confirm each position's baseline/model/hybrid MAE
still matches the locked-in numbers in CLAUDE.md's Verification section.

Checks the OFFICIAL metric: top-40-by-projection pool, single season, which
is the only pool comparable to published accuracy studies. One season only --
this fires on every src/*.py edit, so the full multi-season run is too slow.

Warns loudly (stderr + exit code 2) on any mismatch or failure to run.
Never modifies files -- read-only check.
"""
import json
import os
import re
import subprocess
import sys

# Locked-in MAE per position, mirroring CLAUDE.md's Verification section.
# Top-40 pool, 2023. Anchored per position and comparator so a change in
# print order can never check one against another's expectations.
EXPECTED_MAE = {
    "WR": {"baseline_last4": 7.258, "xgb_model": 6.806, "hybrid": 6.821},
    "RB": {"baseline_last4": 6.129, "xgb_model": 5.746, "hybrid": 5.784},
}

EXCLUDED_BASENAMES = {"LOG.md", "CLAUDE.md", "README.md"}

EVAL_ARGS = ["src/evaluate.py", "--regression"]


def mae_pattern(position, comparator):
    return re.compile(
        rf"^REGRESSION {position} {comparator} MAE:\s*([0-9.]+)", re.MULTILINE
    )


def should_check(file_path, project_dir):
    if not file_path:
        return False
    if os.path.basename(file_path) in EXCLUDED_BASENAMES:
        return False
    if not file_path.endswith(".py"):
        return False

    abs_path = os.path.normcase(os.path.abspath(file_path))
    src_dir = os.path.normcase(os.path.abspath(os.path.join(project_dir, "src")))
    return abs_path.startswith(src_dir + os.sep)


def warn(message):
    banner = "!" * 70
    print(f"\n{banner}\n{message}\n{banner}\n", file=sys.stderr)
    sys.exit(2)


def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {}

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
    tool_input = payload.get("tool_input", {}) or {}
    file_path = tool_input.get("file_path")

    if not should_check(file_path, project_dir):
        sys.exit(0)

    command = [sys.executable] + [os.path.join(project_dir, p) if p.endswith(".py") else p
                                  for p in EVAL_ARGS]
    try:
        result = subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=280,
        )
    except Exception as e:
        warn(f"REGRESSION CHECK COULD NOT RUN\nEdited file: {file_path}\n"
             f"Failed to execute {' '.join(EVAL_ARGS)}: {e}")
        return

    if result.returncode != 0:
        warn(
            "REGRESSION CHECK FAILED TO RUN\n"
            f"Edited file: {file_path}\n"
            f"{' '.join(EVAL_ARGS)} exited with code {result.returncode}\n\n"
            f"stderr:\n{result.stderr.strip()[-2000:]}"
        )
        return

    output = result.stdout
    actual = {}
    missing = []
    for position, expected_metrics in EXPECTED_MAE.items():
        actual[position] = {}
        for metric in expected_metrics:
            match = mae_pattern(position, metric).search(output)
            if match:
                actual[position][metric] = float(match.group(1))
            else:
                missing.append(f"{position} {metric}")

    if missing:
        warn(
            "REGRESSION CHECK COULD NOT PARSE RESULTS\n"
            f"Edited file: {file_path}\n"
            f"Could not find MAE output for: {', '.join(missing)}\n"
            "evaluate.py's output format, comparator ids or ACTIVE_POSITIONS "
            "may have changed.\n\n"
            f"stdout tail:\n{output.strip()[-2000:]}"
        )
        return

    mismatches = []
    for position, expected_metrics in EXPECTED_MAE.items():
        for metric, expected in expected_metrics.items():
            got = actual[position][metric]
            if round(got, 3) != round(expected, 3):
                mismatches.append(f"  {position} {metric}: expected MAE {expected:.3f}, got {got:.3f}")

    if mismatches:
        warn(
            "REGRESSION DETECTED\n"
            f"Edited file: {file_path}\n"
            "Top-40 pool results no longer match the locked-in numbers in CLAUDE.md:\n"
            + "\n".join(mismatches)
            + "\n\nInvestigate before proceeding -- treat unexplained changes (even "
            "improvements) as a leakage suspect, per CLAUDE.md's Verification section."
        )
        return

    summary = "; ".join(
        f"{position} {m['baseline_last4']:.3f}/{m['xgb_model']:.3f}/{m['hybrid']:.3f}"
        for position, m in actual.items()
    )
    print(
        f"Regression check passed (edited {file_path}): top-40 pool "
        f"baseline/model/hybrid MAE {summary} -- all match CLAUDE.md.",
        file=sys.stderr,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
