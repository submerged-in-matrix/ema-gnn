"""
Sweep qe_vcrelax_to_extxyz parsing/dedup logic across all vc-relax .out files
in a directory. Writes one .xyz per composition plus a summary CSV.

Usage:
    python sweep_extract.py <input_dir> [tol] [output_dir]

Assumes each .out file is a QE vc-relax run (same format as fecr_fe08cr08_vcr.out).
"""
import sys
import os
import glob
import csv

# reuse the already-validated single-file parser
from qe_vcrelax_to_extxyz import parse_qe_vcrelax, dedup_frames, write_extxyz


def sweep(input_dir, tol=1e-5, output_dir=None):
    output_dir = output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    out_files = sorted(glob.glob(os.path.join(input_dir, "*.out")))
    if not out_files:
        print(f"No .out files found in {input_dir}")
        return

    summary = []
    total_raw, total_kept = 0, 0

    for path in out_files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            frames, note = parse_qe_vcrelax(path)
            kept = dedup_frames(frames, tol=tol)
            xyz_path = os.path.join(output_dir, name + ".xyz")
            write_extxyz(kept, xyz_path)
            status = "ok" if not note else f"ok ({note})"
            summary.append((name, len(frames), len(kept), status))
            total_raw += len(frames)
            total_kept += len(kept)
            note_str = f"  [{note}]" if note else ""
            print(f"[ok]   {name}: {len(frames)} raw -> {len(kept)} kept{note_str}")
        except Exception as e:
            summary.append((name, 0, 0, f"FAILED: {e}"))
            print(f"[FAIL] {name}: {e}")

    summary_path = os.path.join(output_dir, "sweep_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["composition_file", "raw_frames", "kept_frames", "status"])
        w.writerows(summary)

    print()
    print(f"Files processed : {len(out_files)}")
    print(f"Total raw frames: {total_raw}")
    print(f"Total kept frames: {total_kept}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sweep_extract.py <input_dir> [tol] [output_dir]")
        sys.exit(1)

    input_dir = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    sweep(input_dir, tol=tol, output_dir=output_dir)