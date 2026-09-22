# Video Edit QC Checker

Automatically checks a talking-head video edit for common mistakes:
long pauses, repeated/fluffed lines, sudden audio level jumps, and
possible cut-off words — so you don't have to watch the whole thing
yourself to catch them.

## What it does

1. **Pause detection** — uses ffmpeg's `silencedetect` to flag any
   silence longer than a threshold (default 1.0s).
2. **Audio level jump detection** — measures loudness (RMS dB) in
   rolling windows and flags jumps bigger than a threshold (default 8dB).
3. **Repeated line detection** — transcribes the video with Whisper
   and flags near-duplicate lines spoken close together (a fluffed
   take that got left in).
4. **Possible cut-off word detection** — flags spots where speech
   ends abruptly right before a silence/cut boundary, without normal
   sentence-ending punctuation.

Every detected issue is written to a results file (`results.json`)
with its timestamp, type, and details, so the count and the score
can be recomputed independently — nothing is just claimed.

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) installed and on your PATH
- Python packages:
  ```bash
  pip install openai-whisper numpy
  ```
  (Whisper will download its model the first time it runs — this
  needs an internet connection once, after that it's cached locally.)

## Usage

### 1. Run the QC check on your video

```bash
python qc_check.py --input your_video.mp4 --output results.json
```

Optional flags:
- `--pause-threshold 1.0` — seconds of silence to count as a pause
- `--jump-db 8.0` — dB change to count as a level jump
- `--whisper-model small` — Whisper model size (`tiny`, `base`,
  `small`, `medium`, `large` — bigger is more accurate but slower)
- `--skip-transcript` — skip Whisper entirely and only run pause +
  level-jump detection (fast, no transcript needed)

This prints progress as it runs and writes `results.json` with every
issue found.

### 2. Score it against your own labelled test set

If you've manually logged known mistakes with timestamps (see
`mistakes.csv` for the format), you can compute a real accuracy
score:

```bash
python score_against_log.py --results results.json --log mistakes.csv
```

This matches each logged mistake to the closest tool-detected issue
within a time tolerance (default 2 seconds), and reports:
- how many were caught
- how many were missed
- how many false alarms the tool raised
- an overall accuracy percentage

It writes a full `score_report.json` with the matched/missed/false-alarm
breakdown, so the score is provable and not just a claimed number.

## Mistake log format (`mistakes.csv`)

```
timestamp,type
00:01:28,cough
00:01:46,repeated
00:01:53,pause
...
```

Timestamps can be `HH:MM:SS`, `MM:SS`, or raw seconds.

## Notes on accuracy

- The tool's pause and level-jump detection work directly on the
  audio waveform and don't depend on Whisper, so they're fast and
  reliable.
- Repeated-line and cut-off-word detection depend on Whisper's
  transcript accuracy. Using a larger `--whisper-model` (e.g.
  `medium`) generally improves this at the cost of runtime.
- The default `--tolerance 2.0` in the scorer allows for the fact
  that a manually logged timestamp and the tool's detected timestamp
  won't be pixel-perfect identical. Lower it for a stricter test.

## One-command run

```bash
./run.sh my_video.mp4
```

This installs dependencies (skipped if already present), runs the QC
check, and scores the result against `mistakes.csv`, writing
`results.json` and `score_report.json`.

You can also run each step manually:

```bash
pip install -r requirements.txt
python qc_check.py -i my_video.mp4 -o results.json
python score_against_log.py -r results.json -l mistakes.csv
```
