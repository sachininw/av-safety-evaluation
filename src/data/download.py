"""
Downloads a subset of the Waymo Open Dataset v2 from Google Cloud Storage.

Prerequisites
-------------
1. Accept the Waymo Dataset Terms of Service at:
   https://waymo.com/open/licensing/

2. Authenticate your Google account:
   gcloud auth login

3. Run this script:
   python src/data/download.py --split validation --num_segments 5
"""

import argparse
import subprocess
import sys
from pathlib import Path

GCS_ROOT = "gs://waymo_open_dataset_v_2_0_0"
COMPONENTS = ["lidar_box", "vehicle_pose"]


def run_gsutil(src: str, dst: Path, dry_run: bool = False) -> bool:
    cmd = ["gsutil", "-m", "cp", src, str(dst)]
    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr.strip()}")
        return False
    return True


def list_segment_files(split: str, component: str) -> list[str]:
    """List all parquet files in a given split/component on GCS."""
    gcs_path = f"{GCS_ROOT}/{split}/{component}/"
    result = subprocess.run(
        ["gsutil", "ls", gcs_path], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[ERROR] Cannot list {gcs_path}: {result.stderr.strip()}")
        print("Make sure you have authenticated: gcloud auth login")
        sys.exit(1)
    return [line.strip() for line in result.stdout.splitlines() if line.endswith(".parquet")]


def download(split: str, num_segments: int, out_dir: Path, dry_run: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for component in COMPONENTS:
        files = list_segment_files(split, component)
        files = files[:num_segments]

        component_dir = out_dir / split / component
        component_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading {len(files)} files for {split}/{component} ...")
        for gcs_path in files:
            fname = Path(gcs_path).name
            dst = component_dir / fname
            if dst.exists():
                print(f"  [SKIP] {fname} already exists")
                continue
            print(f"  {fname}")
            run_gsutil(gcs_path, dst, dry_run=dry_run)

    print("\nDownload complete.")
    print(f"Data saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Waymo Open Dataset v2 samples")
    parser.add_argument("--split", default="validation", choices=["training", "validation", "testing"])
    parser.add_argument("--num_segments", type=int, default=5, help="Number of segments per component to download")
    parser.add_argument("--out_dir", default="data", help="Local output directory")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    download(
        split=args.split,
        num_segments=args.num_segments,
        out_dir=Path(args.out_dir),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
