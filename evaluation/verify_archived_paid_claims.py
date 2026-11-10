#!/usr/bin/env python3
"""Offline check of the archived "paid" sentiment results (no network, no API calls).

Commit 7688fa6 carried evaluation/sentiment_eval_results.json labelled
``mode: paid``. This script reads it from git history (read-only), regenerates
the rows with the offline dry-run simulator exactly as it was at that commit
(``git archive`` into a temp dir), and reports:

  A. whether the archived rows are reproduced exactly by the simulator;
  B. whether the rows carry any provider usage (tokens, latency) as the summary claims;
  C. whether the summary's Macro-F1 can be reconstructed from the rows;
  D. whether any provider log or raw response file was ever committed.

Run from the repository root:  python evaluation/verify_archived_paid_claims.py
Exit code 0 = all three findings confirmed, 1 = something differs, 2 = history unavailable.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ARCHIVE_COMMIT = "7688fa6"
ARCHIVED_PATH = "evaluation/sentiment_eval_results.json"
ROW_FIELDS = (
    "score",
    "confidence",
    "predicted_direction",
    "gate_action",
    "key_positive",
    "key_negative",
    "summary",
)
EVIDENCE_NAME = re.compile(r"(\.log$|raw|response|usage|transcript|task-\d+)", re.I)
ROOT = Path(__file__).resolve().parent.parent


def git(*args: str, binary: bool = False):
    out = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return out if binary else out.decode()


def main() -> int:
    try:
        git("cat-file", "-e", f"{ARCHIVE_COMMIT}^{{commit}}")
    except subprocess.CalledProcessError:
        print(
            f"commit {ARCHIVE_COMMIT} not in this clone (shallow?); fetch full history to run this check"
        )
        return 2

    archived = json.loads(git("show", f"{ARCHIVE_COMMIT}:{ARCHIVED_PATH}"))
    a_rows = {
        pt: {r["example_id"]: r for r in rows}
        for pt, rows in archived["detailed_results"].items()
    }
    a_sum = archived["summary"]

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(
            fileobj=io.BytesIO(git("archive", ARCHIVE_COMMIT, binary=True))
        ) as tar:
            tar.extractall(tmp, filter="data")
        out = Path(tmp) / "dry.json"
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OPENAI_API_KEY", "ALPHASIGNAL_API_KEY")
        }
        proc = subprocess.run(
            [
                sys.executable,
                "evaluation/sentiment_eval_runner.py",
                "--dry-run",
                "--output",
                str(out),
            ],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(
                "dry-run simulator failed at the archived commit:\n"
                + proc.stdout[-2000:]
                + proc.stderr[-4000:]
            )
            return 2
        sim = json.loads(out.read_text())
    s_rows = {
        pt: {r["example_id"]: r for r in rows}
        for pt, rows in sim["detailed_results"].items()
    }
    s_sum = sim["summary"]

    total = same = 0
    for pt, rows in a_rows.items():
        for eid, row in rows.items():
            total += 1
            same += all(row[f] == s_rows[pt][eid][f] for f in ROW_FIELDS)
    a_ok = total > 0 and same == total

    ptok = sum(r["prompt_tokens"] for rows in a_rows.values() for r in rows.values())
    ctok = sum(
        r["completion_tokens"] for rows in a_rows.values() for r in rows.values()
    )
    lat = sum(r["latency_s"] for rows in a_rows.values() for r in rows.values())
    claimed = a_sum["budget_usage"]
    b_ok = ptok == 0 and ctok == 0 and lat == 0 and claimed["total_prompt_tokens"] > 0

    print(
        f"archived file: {ARCHIVE_COMMIT}:{ARCHIVED_PATH}   summary mode={a_sum['mode']!r} model={a_sum['model']!r}"
    )
    print(
        f"A. rows reproduced exactly by the dry-run simulator: {same}/{total} rows identical on {len(ROW_FIELDS)} fields"
    )
    print(
        f"B. usage in rows: prompt={ptok} completion={ctok} latency_s={lat}  | summary claims prompt={claimed['total_prompt_tokens']}"
        f" completion={claimed['total_completion_tokens']} cost=${claimed['total_cost_usd']} requests={claimed['total_requests']}"
    )

    c_ok = False
    print("C. Macro-F1 (recomputed from the identical rows)  vs  summary claim")
    mismatches = 0
    for split in ("dev_metrics", "holdout_metrics"):
        for pt in ("baseline", "candidate"):
            claimed_f1 = a_sum[split][pt]["macro_f1"]
            recomputed = s_sum[split][pt]["macro_f1"]
            mismatches += claimed_f1 != recomputed
            print(
                f"   {split[:-8]:<8} {pt:<9} recomputed={recomputed:<7} claimed={claimed_f1}"
            )
    c_ok = mismatches > 0 and a_ok

    names = sorted(
        {
            n
            for n in git("log", "--all", "--name-only", "--pretty=format:").splitlines()
            if n and EVIDENCE_NAME.search(n)
        }
    )
    d_ok = not names
    print(
        f"D. provider log / raw response / usage files ever committed: {names or 'none'}"
    )

    print()
    print(f"FINDING CONFIRMED: archived rows are simulator output ......... {a_ok}")
    print(f"FINDING CONFIRMED: rows carry no provider usage ............... {b_ok}")
    print(f"FINDING CONFIRMED: reported F1 not reconstructable from rows .. {c_ok}")
    print(f"No retained provider evidence in git history .................. {d_ok}")
    return 0 if (a_ok and b_ok and c_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
