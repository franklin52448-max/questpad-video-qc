#!/usr/bin/env python3
"""
Video Edit QC Checker
======================
Takes a talking-head mp4 and flags likely editing mistakes:
  - Long pauses / silences (> configurable threshold, default 1.0s)
  - Repeated / duplicated lines (via transcript similarity)
  - Abrupt audio level jumps between segments
  - Possible cut-off words (silence starting/ending mid-word based on
    transcript word timestamps)

Writes a JSON results file with every detected issue + timestamp,
so the score can be recomputed independently from the file itself.

Usage:
    python qc_check.py --input video.mp4 --output results.json

Requires:
    ffmpeg / ffprobe on PATH
    Python packages: openai-whisper, numpy
    (pip install openai-whisper numpy)
"""

import argparse
import json
import subprocess
import sys
import tempfile
import os
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Audio extraction
# ----------------------------------------------------------------------

def extract_audio(video_path: str, out_wav: str, sample_rate: int = 16000):
    """Extract mono 16kHz WAV audio from the input video via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(sample_rate),
        "-vn", out_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")


def get_duration_seconds(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    return float(result.stdout.strip())


# ----------------------------------------------------------------------
# Silence / pause detection (ffmpeg silencedetect)
# ----------------------------------------------------------------------

def detect_silences(video_path: str, noise_db: str = "-30dB", min_dur: float = 1.0):
    """
    Uses ffmpeg's silencedetect filter to find silences >= min_dur seconds.
    Returns a list of dicts: {"start": float, "end": float, "duration": float}
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-af", f"silencedetect=noise={noise_db}:d={min_dur}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    silences = []
    start = None
    for line in stderr.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                start = float(line.split("silence_start:")[1].strip())
            except ValueError:
                start = None
        elif "silence_end:" in line and start is not None:
            # format: silence_end: 12.34 | silence_duration: 1.23
            try:
                end_part = line.split("silence_end:")[1].strip()
                end_str = end_part.split("|")[0].strip()
                end = float(end_str)
                dur = end - start
                silences.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "duration": round(dur, 2),
                })
            except (ValueError, IndexError):
                pass
            start = None
    return silences


# ----------------------------------------------------------------------
# Audio level jump detection (RMS per fixed-size window)
# ----------------------------------------------------------------------

def read_wav_samples(wav_path: str):
    """Read a mono 16-bit PCM WAV file into a numpy float array in [-1, 1]."""
    import wave
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise RuntimeError("Expected 16-bit PCM WAV audio.")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, framerate


def detect_level_jumps(wav_path: str, window_sec: float = 3.0, jump_db: float = 8.0):
    """
    Splits audio into fixed windows, computes RMS dB per window, and flags
    any window-to-window jump greater than jump_db as a possible level jump.
    Returns list of {"time": float, "from_db": float, "to_db": float, "jump_db": float}
    """
    samples, sr = read_wav_samples(wav_path)
    window_len = int(window_sec * sr)
    if window_len <= 0:
        return []

    n_windows = len(samples) // window_len
    rms_db = []
    for i in range(n_windows):
        seg = samples[i * window_len:(i + 1) * window_len]
        rms = np.sqrt(np.mean(seg ** 2)) + 1e-9
        db = 20 * np.log10(rms)
        rms_db.append(db)

    jumps = []
    last_audible_idx = None
    for i in range(len(rms_db)):
        if rms_db[i] < -50:
            # silent window (likely a pause) - don't compare across it,
            # but don't reset last_audible_idx either, so we compare the
            # next audible window against the last audible one before the gap
            continue
        if last_audible_idx is not None:
            delta = rms_db[i] - rms_db[last_audible_idx]
            if abs(delta) >= jump_db:
                jumps.append({
                    "time": round(i * window_sec, 2),
                    "from_db": round(rms_db[last_audible_idx], 1),
                    "to_db": round(rms_db[i], 1),
                    "jump_db": round(delta, 1),
                })
        last_audible_idx = i
    return jumps


# ----------------------------------------------------------------------
# Transcript-based detection (Whisper): repeated lines, cut-off words
# ----------------------------------------------------------------------

def transcribe(video_path: str, model_size: str = "small"):
    """
    Runs Whisper with word-level timestamps.
    Returns the whisper result dict (segments with words).
    """
    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )

    model = whisper.load_model(model_size)
    result = model.transcribe(video_path, word_timestamps=True, verbose=False)
    return result


def _normalize(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalnum() or c.isspace()).strip()


def detect_repeated_lines(segments, min_words: int = 4, similarity_threshold: float = 0.85):
    """
    Looks for near-duplicate consecutive-ish segments (fluffed lines that
    were repeated). Uses simple token overlap as a similarity proxy so this
    has no extra dependencies.
    Returns list of {"time": float, "text": str, "repeats_time": float, "repeats_text": str}
    """
    def token_set(t):
        return set(_normalize(t).split())

    flagged = []
    n = len(segments)
    for i in range(n):
        seg_a = segments[i]
        text_a = seg_a.get("text", "")
        if len(text_a.split()) < min_words:
            continue
        set_a = token_set(text_a)
        # only compare to a nearby window (repeats happen close together)
        for j in range(i + 1, min(i + 6, n)):
            seg_b = segments[j]
            text_b = seg_b.get("text", "")
            if len(text_b.split()) < min_words:
                continue
            set_b = token_set(text_b)
            if not set_a or not set_b:
                continue
            overlap = len(set_a & set_b) / max(len(set_a), len(set_b))
            if overlap >= similarity_threshold:
                flagged.append({
                    "time": round(seg_a["start"], 2),
                    "text": text_a.strip(),
                    "repeats_time": round(seg_b["start"], 2),
                    "repeats_text": text_b.strip(),
                })
                break
    return flagged


def detect_possible_cutoffs(segments, silences, gap_threshold: float = 0.15):
    """
    Heuristic: if a segment ends very close to (within gap_threshold seconds
    before) a detected silence/cut boundary, and the last word looks
    incomplete (no ending punctuation and short trailing word), flag it as a
    possible cut-off word.
    Returns list of {"time": float, "text": str}
    """
    flagged = []
    silence_starts = [s["start"] for s in silences]
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        end = seg["end"]
        for s_start in silence_starts:
            if 0 <= (s_start - end) <= gap_threshold:
                if not text.endswith((".", "!", "?", ",")):
                    flagged.append({"time": round(end, 2), "text": text[-40:]})
                break
    return flagged


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def format_ts(seconds: float) -> str:
    seconds = max(0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def main():
    parser = argparse.ArgumentParser(description="QC-check a talking head video edit.")
    parser.add_argument("--input", "-i", required=True, help="Path to input mp4")
    parser.add_argument("--output", "-o", default="results.json", help="Path to write results JSON")
    parser.add_argument("--pause-threshold", type=float, default=1.0, help="Min silence duration (s) to flag as a pause")
    parser.add_argument("--jump-db", type=float, default=8.0, help="dB delta to flag as a level jump")
    parser.add_argument("--whisper-model", default="small", help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--skip-transcript", action="store_true", help="Skip Whisper transcript checks (pauses + level jumps only)")
    args = parser.parse_args()

    video_path = args.input
    if not os.path.exists(video_path):
        print(f"ERROR: input file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/5] Reading video info: {video_path}")
    duration = get_duration_seconds(video_path)
    print(f"      Duration: {format_ts(duration)} ({duration:.1f}s)")

    print(f"[2/5] Detecting silences/pauses (>= {args.pause_threshold}s)...")
    silences = detect_silences(video_path, min_dur=args.pause_threshold)
    print(f"      Found {len(silences)} pause(s).")

    print("[3/5] Extracting audio + checking for level jumps...")
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, wav_path)
        jumps = detect_level_jumps(wav_path, jump_db=args.jump_db)
    print(f"      Found {len(jumps)} level jump(s).")

    repeats = []
    cutoffs = []
    if not args.skip_transcript:
        print(f"[4/5] Transcribing with Whisper ({args.whisper_model} model)... this can take a while.")
        try:
            whisper_result = transcribe(video_path, model_size=args.whisper_model)
            segments = whisper_result.get("segments", [])
            print(f"      Transcribed {len(segments)} segments.")
            repeats = detect_repeated_lines(segments)
            cutoffs = detect_possible_cutoffs(segments, silences)
            print(f"      Found {len(repeats)} possible repeated line(s).")
            print(f"      Found {len(cutoffs)} possible cut-off word(s).")
        except RuntimeError as e:
            print(f"      WARNING: transcript checks skipped: {e}")
    else:
        print("[4/5] Skipping transcript checks (--skip-transcript).")

    print("[5/5] Writing results...")

    issues = []
    for s in silences:
        issues.append({
            "type": "pause",
            "time": s["start"],
            "time_formatted": format_ts(s["start"]),
            "duration_sec": s["duration"],
            "detail": f"Silence of {s['duration']}s starting at {format_ts(s['start'])}",
        })
    for j in jumps:
        issues.append({
            "type": "level_jump",
            "time": j["time"],
            "time_formatted": format_ts(j["time"]),
            "jump_db": j["jump_db"],
            "detail": f"Audio level changed by {j['jump_db']}dB ({j['from_db']}dB -> {j['to_db']}dB) around {format_ts(j['time'])}",
        })
    for r in repeats:
        issues.append({
            "type": "repeated_line",
            "time": r["time"],
            "time_formatted": format_ts(r["time"]),
            "text": r["text"],
            "repeats_time": r["repeats_time"],
            "detail": f"Line at {format_ts(r['time'])} appears repeated at {format_ts(r['repeats_time'])}: \"{r['text']}\"",
        })
    for c in cutoffs:
        issues.append({
            "type": "possible_cutoff",
            "time": c["time"],
            "time_formatted": format_ts(c["time"]),
            "text": c["text"],
            "detail": f"Possible cut-off word near {format_ts(c['time'])}: \"...{c['text']}\"",
        })

    issues.sort(key=lambda x: x["time"])

    output = {
        "input_file": os.path.basename(video_path),
        "duration_sec": round(duration, 2),
        "duration_formatted": format_ts(duration),
        "settings": {
            "pause_threshold_sec": args.pause_threshold,
            "level_jump_db_threshold": args.jump_db,
            "whisper_model": None if args.skip_transcript else args.whisper_model,
        },
        "summary": {
            "total_issues": len(issues),
            "pauses": len(silences),
            "level_jumps": len(jumps),
            "repeated_lines": len(repeats),
            "possible_cutoffs": len(cutoffs),
        },
        "issues": issues,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. {len(issues)} total issue(s) written to {args.output}")
    print(f"  Pauses:          {len(silences)}")
    print(f"  Level jumps:     {len(jumps)}")
    print(f"  Repeated lines:  {len(repeats)}")
    print(f"  Possible cutoffs:{len(cutoffs)}")


if __name__ == "__main__":
    main()
