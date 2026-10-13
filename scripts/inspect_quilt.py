"""One-off Quilt-1M exploration: CSV head and zip contents.

Usage:
    python scripts/inspect_quilt.py --csv path/to/quilt_1M_lookup.csv --zip_dir path/to/zips
"""

import argparse
import glob
import os
import zipfile

import pandas as pd

parser = argparse.ArgumentParser(description="Inspect Quilt-1M CSV and image zips")
parser.add_argument("--csv", required=True, help="Path to quilt_1M_lookup.csv")
parser.add_argument("--zip_dir", required=True, help="Directory with images_part_*.zip")
args = parser.parse_args()

print("=" * 60)
print("1a) CSV YAPISI")
print("=" * 60)
df = pd.read_csv(args.csv, nrows=5)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)
print("Sutunlar:", list(df.columns))
print()
print(df.to_string())
print()
print(df.dtypes)

print()
print("=" * 60)
print("1b) ZIP ICERIGI (images_part_1.zip - ilk 20)")
print("=" * 60)
zip_path = os.path.join(args.zip_dir, "images_part_1.zip")
with zipfile.ZipFile(zip_path, "r") as z:
    names = z.namelist()
    print(f"Bu ZIP'teki toplam dosya: {len(names)}")
    for n in names[:20]:
        print(" ", n)
    exts = {n.rsplit(".", 1)[-1].lower() for n in names if "." in n}
    print(f"Uzantilar: {exts}")

print()
print("=" * 60)
print("1c) TUM ZIP DOSYALARI")
print("=" * 60)
zips = sorted(glob.glob(os.path.join(args.zip_dir, "images_part_*.zip")))
print(f"Toplam ZIP: {len(zips)}")
for z in zips:
    print(f"  {os.path.basename(z)}  -  {os.path.getsize(z) / 1024 / 1024:.0f} MB")
