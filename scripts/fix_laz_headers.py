"""Fix nonstandard LAZ headers so lidR can read them."""
from pathlib import Path
import laspy
import numpy as np

SRC_DIR = Path("/mnt/d/TopTreeSegR/laz")
DST_DIR = Path("/mnt/d/TopTreeSegR/laz_fixed")
DST_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "Rem_Piensk_2016_0901202.laz",
    "Rem_Piensk_2016_2400402.laz",
    "Rem_Piensk_2016_3100502.laz",
    "Rem_Suprasl_2015_2402904.laz",
]

for fname in FILES:
    src = SRC_DIR / fname
    dst = DST_DIR / fname
    if not src.exists():
        print(f"MISSING: {fname}")
        continue

    las = laspy.read(src)
    print(f"{fname}: {len(las.points):,} points, scales {las.header.scales}")

    las.header.scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    las.header.offsets = np.array(
        [las.x.min(), las.y.min(), las.z.min()], dtype=np.float64
    )

    las.write(dst)
    print(f"  -> fixed: {dst}")