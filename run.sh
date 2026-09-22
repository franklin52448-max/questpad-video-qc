#!/usr/bin/env bash
# One-command runner: installs deps (if needed), runs the QC check on the
# given video, then scores it against mistakes.csv.
#
# Usage: ./run.sh your_video.mp4

set -e

if [ -z "$1" ]; then
  echo "Usage: ./run.sh path/to/video.mp4"
  exit 1
fi

VIDEO="$1"

echo "== Installing dependencies (skips if already installed) =="
pip install -q -r requirements.txt

echo ""
echo "== Running QC check on: $VIDEO =="
python3 qc_check.py --input "$VIDEO" --output results.json

echo ""
echo "== Scoring against mistakes.csv =="
python3 score_against_log.py --results results.json --log mistakes.csv --output score_report.json

echo ""
echo "Done. See results.json and score_report.json"
