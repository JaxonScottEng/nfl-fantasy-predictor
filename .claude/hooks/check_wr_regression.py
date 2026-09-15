#!/usr/bin/env python
"""
PostToolUse hook: after an edit to a .py file under src/, re-run
src/train.py and confirm each position's baseline/model/hybrid MAE values
still match the locked-in numbers in CLAUDE.md's Verification section.

Warns loudly (stderr + exit code 2) on any mismatch or failure to run.
Never modifies files -- read-only check.
"""
import json
import os
import re
import subprocess
import sys

# Locked-in MAE per position, mirroring CLAUDE.md's Verification section.
# Anchored per position so a change in print order can never make one
# position's numbers get checked against another's expectations.
EXPECTED_MAE = {
    "WR": {"Baseline": 4.677, "Model": 4.543, "Hybrid": 4.492},
    "RB": {"Baseline": 4.543, "Model": 4.489, "Hybrid": 4.390},
}

EXCLUDED_BASENAMES = {"LOG.md", "CLAUDE.md", "README.md"}

LINE_LABELS = {
    "Baseline": r"Baseline \(last-4 avg\)",
    "Model": r"Model \(XGBoost\)",
    "Hybrid": r"Hybrid \(segmented\)",
}


def mae_pattern(position, metric):
    return re.compile(rf"^{position} {LINE_LABELS[metric]}\s+MAE:\s*([0-9.]+)", re.MULTILINE)


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

    train_script = os.path.join(project_dir, "src", "train.py")
    try:
        result = subprocess.run(
            [sys.executable, train_script],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=280,
        )
    except Exception as e:
        warn(f"WR REGRESSION CHECK COULD NOT RUN\nEdited file: {file_path}\nFailed to execute src/train.py: {e}")
        return

    if result.returncode != 0:
        warn(
            "WR REGRESSION CHECK FAILED TO RUN\n"
            f"Edited file: {file_path}\n"
            f"src/train.py exited with code {result.returncode}\n\n"
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
            "train.py's output format or ACTIVE_POSITIONS may have changed.\n\n"
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
            "src/train.py no longer matches the locked-in results in CLAUDE.md:\n"
            + "\n".join(mismatches)
            + "\n\nInvestigate before proceeding -- treat unexplained changes (even "
            "improvements) as a leakage suspect, per CLAUDE.md's Verification section."
        )
        return

    summary = "; ".join(
        f"{position} {m['Baseline']:.3f}/{m['Model']:.3f}/{m['Hybrid']:.3f}"
        for position, m in actual.items()
    )
    print(
        f"Regression check passed (edited {file_path}): "
        f"baseline/model/hybrid MAE {summary} -- all match CLAUDE.md.",
        file=sys.stderr,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
