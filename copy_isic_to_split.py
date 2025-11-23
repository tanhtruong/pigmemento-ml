import csv
import shutil
from pathlib import Path

# ====== CONFIG ======

CSV_PATH = "val_metadata.csv"  # your ISIC metadata file

SOURCE_FOLDER = Path("data_raw")  # all images live here
TARGET_ROOT = Path("data")        # base folder for train/val/test

IMAGE_EXT = ".jpg"                # adjust if needed

DEST_SPLIT = "val"              # <<< CHANGE ME: train, val or test

# ====================


def row_to_label(row):
    """
    Label rule:
      If diagnosis_1 contains "benign" (case-insensitive) -> benign
      Else -> malignant
    """
    diag1 = (row.get("diagnosis_1") or "").strip().lower()

    if "benign" in diag1:
        return "benign"
    else:
        return "malignant"


def ensure_dirs():
    for label in ["benign", "malignant"]:
        path = TARGET_ROOT / DEST_SPLIT / label
        path.mkdir(parents=True, exist_ok=True)


def main():
    csv_path_obj = Path(CSV_PATH)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path_obj}")

    ensure_dirs()

    copied = 0
    missing = 0

    with open(csv_path_obj, "r", newline="") as f:
        reader = csv.DictReader(f)

        if "isic_id" not in reader.fieldnames:
            raise ValueError("CSV must contain 'isic_id' column.")
        if "diagnosis_1" not in reader.fieldnames:
            raise ValueError("CSV must contain 'diagnosis_1' column.")

        for row in reader:
            isic_id = (row.get("isic_id") or "").strip()
            if not isic_id:
                continue

            filename = isic_id + IMAGE_EXT
            label = row_to_label(row)

            src = SOURCE_FOLDER / filename
            dst = TARGET_ROOT / DEST_SPLIT / label / filename

            if not src.exists():
                print(f"WARNING: missing file: {src}")
                missing += 1
                continue

            shutil.copy2(src, dst)
            copied += 1

    print("\n===== DONE =====")
    print(f"Destination split: {DEST_SPLIT}")
    print(f"Copied images:     {copied}")
    print(f"Missing images:    {missing}")
    print(f"Output in:         {TARGET_ROOT.resolve()}")


if __name__ == "__main__":
    main()