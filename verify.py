import csv
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: Install OpenCV for Python: pip install opencv-python")
    sys.exit(1)

if len(sys.argv) != 3:
    print("Usage: python verify.py <images_per_class> <threads>")
    print("Example: python verify.py 50 4")
    sys.exit(1)

count = int(sys.argv[1])
threads = int(sys.argv[2])

root = Path(__file__).resolve().parent

seq_csv = root / "results" / f"sequential_{count}_per_class.csv"
par_csv = root / "results" / f"parallel_{count}_t{threads}_per_class.csv"

seq_root = root / "output" / "reconstructed" / str(count)
par_root = (
    root / "output" / "parallel_reconstructed"
    / str(count) / f"t{threads}"
)

if not seq_csv.exists():
    print("ERROR: Sequential CSV not found:", seq_csv)
    sys.exit(1)

if not par_csv.exists():
    print("ERROR: Parallel CSV not found:", par_csv)
    sys.exit(1)

def read_results(path):
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Class"] == "OVERALL":
                continue
            key = (row["Class"], row["Image"])
            data[key] = row
    return data

seq = read_results(seq_csv)
par = read_results(par_csv)

print("Sequential image results:", len(seq))
print("Parallel image results  :", len(par))

missing = sorted(set(seq) - set(par))
extra = sorted(set(par) - set(seq))

if missing:
    print("Missing parallel results:", missing[:10])

if extra:
    print("Extra parallel results:", extra[:10])

common = sorted(set(seq) & set(par))

metric_columns = [
    "Compression_Ratio",
    "Space_Saving_Percent",
    "MSE",
    "PSNR_dB",
    "SSIM"
]

metric_ok = True
for key in common:
    for col in metric_columns:
        a = float(seq[key][col])
        b = float(par[key][col])

        if np.isinf(a) and np.isinf(b):
            continue

        if not np.isclose(a, b, rtol=1e-5, atol=1e-5):
            print(
                f"Metric mismatch: {key}, {col}: "
                f"sequential={a}, parallel={b}"
            )
            metric_ok = False

image_ok = True
checked_images = 0

for class_name, image_name in common:
    seq_image = seq_root / class_name / image_name
    par_image = par_root / class_name / image_name

    if not seq_image.exists():
        print("Missing sequential image:", seq_image)
        image_ok = False
        continue

    if not par_image.exists():
        print("Missing parallel image:", par_image)
        image_ok = False
        continue

    a = cv2.imread(str(seq_image), cv2.IMREAD_COLOR)
    b = cv2.imread(str(par_image), cv2.IMREAD_COLOR)

    if a is None or b is None:
        print("Could not read:", class_name, image_name)
        image_ok = False
        continue

    if a.shape != b.shape:
        print("Shape mismatch:", class_name, image_name)
        image_ok = False
        continue

    if not np.array_equal(a, b):
        max_diff = np.max(
            np.abs(a.astype(np.int16) - b.astype(np.int16))
        )
        print(
            f"Pixel mismatch: {class_name}/{image_name}, "
            f"max difference={max_diff}"
        )
        image_ok = False

    checked_images += 1

correct = (
    not missing
    and not extra
    and metric_ok
    and image_ok
    and len(seq) == len(par)
)

print("\n============================================")
print("Correctness Verification")
print("============================================")
print("Images compared :", checked_images)
print("Same image set  :", not missing and not extra)
print("Metrics match   :", metric_ok)
print("Images match    :", image_ok)
print("STATUS          :", "PASS" if correct else "FAIL")
print("============================================")

sys.exit(0 if correct else 1)
