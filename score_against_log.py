#!/usr/bin/env python3
"""
Compares the QC tool's results.json against a manually labelled mistake log
(CSV) to compute a real, provable accuracy score.

Mistake log CSV format (no header needed, but a header row is fine too):
    timestamp,type
    00:01:28,cough
    00:01:46,repeated
    ...

Timestamps can be HH:MM:SS, MM:SS, or raw seconds.

A tool-detected issue "catches" a logged mistake if the tool's timestamp is
within --tolerance seconds (default 2.0s) of the logged timestamp.

Usage:
    python score_against_log.py --results results.json --log mistakes.csv --tolerance 2.0
"""

import argparse
import csv
import json
import sys


def parse_timestamp(ts: str) -> float:
    ts = ts.strip()
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = parts
        return m * 60 + s
    elif len(parts) == 1:
        return parts[0]
    else:
        raise ValueError(f"Cannot parse timestamp: {ts}")


def load_log(path: str):
    entries = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # skip header if first cell isn't a parseable timestamp
    start_idx = 0
    if rows:
        try:
            parse_timestamp(rows[0][0])
        except ValueError:
            start_idx = 1
    for row in rows[start_idx:]:
        if not row or not row[0].strip():
            continue
        ts = parse_timestamp(row[0])
        label = row[1].strip() if len(row) > 1 else "unspecified"
        entries.append({"time": ts, "type": label})
    return entries


def load_results(path: str):
    with open(path) as f:
        data = json.load(f)
    return data.get("issues", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", "-r", required=True, help="Path to results.json from qc_check.py")
    parser.add_argument("--log", "-l", required=True, help="Path to your manual mistake log CSV")
    parser.add_argument("--tolerance", "-t", type=float, default=2.0, help="Seconds of slack when matching timestamps")
    parser.add_argument("--output", "-o", default="score_report.json", help="Where to write the score report")
    args = parser.parse_args()

    log_entries = load_log(args.log)
    issues = load_results(args.results)

    matched = []
    missed = []
    used_issue_idx = set()

    for entry in log_entries:
        best_idx = None
        best_diff = None
        for idx, issue in enumerate(issues):
            if idx in used_issue_idx:
                continue
            diff = abs(issue["time"] - entry["time"])
            if diff <= args.tolerance and (best_diff is None or diff < best_diff):
                best_diff = diff
                best_idx = idx
        if best_idx is not None:
            used_issue_idx.add(best_idx)
            matched.append({
                "logged_time": entry["time"],
                "logged_type": entry["type"],
                "tool_time": issues[best_idx]["time"],
                "tool_type": issues[best_idx]["type"],
                "diff_sec": round(best_diff, 2),
            })
        else:
            missed.append(entry)

    false_alarms = [issues[i] for i in range(len(issues)) if i not in used_issue_idx]

    total_logged = len(log_entries)
    caught = len(matched)
    accuracy_pct = round(100 * caught / total_logged, 2) if total_logged else 0.0

    report = {
        "total_logged_mistakes": total_logged,
        "caught": caught,
        "missed": len(missed),
        "accuracy_pct": accuracy_pct,
        "false_alarms": len(false_alarms),
        "tolerance_sec": args.tolerance,
        "matched_detail": matched,
        "missed_detail": missed,
        "false_alarm_detail": false_alarms,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Logged mistakes: {total_logged}")
    print(f"Caught by tool:  {caught}")
    print(f"Missed:          {len(missed)}")
    print(f"False alarms:    {len(false_alarms)}")
    print(f"Accuracy:        {accuracy_pct}%")
    print(f"\nFull report written to {args.output}")

    if missed:
        print("\nMissed mistakes:")
        for m in missed:
            mm = int(m["time"] // 60)
            ss = m["time"] % 60
            print(f"  {mm:02d}:{ss:05.2f}  {m['type']}")


if __name__ == "__main__":
    main()
