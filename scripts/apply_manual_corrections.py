"""
apply_manual_corrections.py
────────────────────────────
Propagate manually-reviewed corrections from manual_check_annotation_5000.xlsx
back into the two transcript files it was originally built from:

  transcript_hira_kata_rev.txt  (hira_kata_text column)
  transcript_phone3_ojt.txt     (phoneme_text column)

Only rows with check == "v" (reviewed) are considered, and a line is only
rewritten if the xlsx text differs from what's currently in the .txt file.
Rows with a blank check ("need manual check") are left untouched and
reported as skipped.

Usage:
    python scripts/apply_manual_corrections.py
"""
from __future__ import annotations

import datetime
import shutil
from pathlib import Path

import pandas as pd

BASIC5000_DIR = Path("/home/hainas-1/ong-ru/jsut_ver1.1/basic5000")
XLSX_PATH = BASIC5000_DIR / "manual_check_annotation_5000.xlsx"
HIRA_KATA_PATH = BASIC5000_DIR / "transcript_hira_kata_rev.txt"
PHONE3_PATH = BASIC5000_DIR / "transcript_phone3_ojt.txt"
ARCHIVE_DIR = BASIC5000_DIR / "archive"

TRAINING_PHONE3_COPY = (
    Path("/home/hainas-1/ong-ru/ASR_JA_Vowel_Devoicing")
    / "dataset" / "jsut_ver1.1" / "basic5000" / "transcript_phone3_ojt.txt"
)


def load_transcript(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        utt_id, text = line.split(":", 1)
        entries[utt_id] = text
    return entries


def write_transcript(path: Path, order: list[str], entries: dict[str, str]) -> None:
    lines = [f"{utt_id}:{entries[utt_id]}" for utt_id in order]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def backup(path: Path, stamp: str) -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dst = ARCHIVE_DIR / f"{path.name}.bak_{stamp}"
    shutil.copy2(path, dst)
    return dst


def main() -> None:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_excel(XLSX_PATH)

    hira_kata = load_transcript(HIRA_KATA_PATH)
    phone3 = load_transcript(PHONE3_PATH)
    order = list(hira_kata.keys())
    assert set(order) == set(phone3) == set(df["utterance_id"]), \
        "utterance_id sets differ between xlsx and transcript files"

    b1 = backup(HIRA_KATA_PATH, stamp)
    b2 = backup(PHONE3_PATH, stamp)
    print(f"Backed up:\n  {b1}\n  {b2}")

    changed_hira_kata: list[str] = []
    changed_phone3: list[str] = []
    checked_unchanged = 0
    skipped: list[str] = []

    for row in df.itertuples(index=False):
        utt_id = row.utterance_id
        if row.check != "v":
            skipped.append(utt_id)
            continue

        any_change_this_row = False

        if row.hira_kata_text != hira_kata[utt_id]:
            hira_kata[utt_id] = row.hira_kata_text
            changed_hira_kata.append(utt_id)
            any_change_this_row = True

        if row.phoneme_text != phone3[utt_id]:
            phone3[utt_id] = row.phoneme_text
            changed_phone3.append(utt_id)
            any_change_this_row = True

        if not any_change_this_row:
            checked_unchanged += 1

    write_transcript(HIRA_KATA_PATH, order, hira_kata)
    write_transcript(PHONE3_PATH, order, phone3)
    shutil.copy2(PHONE3_PATH, TRAINING_PHONE3_COPY)

    total = len(df)
    checked = total - len(skipped)

    print(f"\nTotal rows:                    {total}")
    print(f"Checked (check == 'v'):         {checked}")
    print(f"Skipped (need manual check):   {len(skipped)}")
    print(f"Checked rows, no diff found:   {checked_unchanged}")
    print(f"Lines changed in hira_kata_rev: {len(changed_hira_kata)}")
    print(f"Lines changed in phone3_ojt:    {len(changed_phone3)}")
    print(f"\nSynced training copy -> {TRAINING_PHONE3_COPY}")

    print("\nSkipped utterance_ids:")
    print(", ".join(skipped))

    print("\nChanged in hira_kata_rev:")
    print(", ".join(changed_hira_kata) if changed_hira_kata else "(none)")

    print("\nChanged in phone3_ojt:")
    print(", ".join(changed_phone3) if changed_phone3 else "(none)")


if __name__ == "__main__":
    main()
