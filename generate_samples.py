#!/usr/bin/env python3
"""
generate_samples.py
Extract lightweight sample datasets from large CSV and Parquet files
for version control tracking.
"""

from pathlib import Path
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def sample_csv(source_path: Path, dest_path: Path, n_rows: int = 1000) -> None:
    """Read the first n rows of a CSV file and export the sample."""
    print(f"Sampling CSV: {source_path.name} -> {dest_path.name}")
    df_sample = pd.read_csv(source_path, nrows=n_rows)
    df_sample.to_csv(dest_path, index=False)


def sample_parquet(source_path: Path, dest_path: Path, n_rows: int = 1000) -> None:
    """Stream rows from a Parquet dataset to prevent out-of-memory errors."""
    print(f"Sampling Parquet: {source_path.name} -> {dest_path.name}")
    dataset = ds.dataset(source_path, format="parquet")
    
    # Read only the required row slice
    table_sample = dataset.head(n_rows)
    pq.write_table(table_sample, dest_path, compression="snappy")


def process_directory(
    raw_dir: Path,
    sample_dir: Path,
    sample_size: int = 1000,
    prefix: str = "sample_"
) -> None:
    """Iterate over all CSV and Parquet files in the raw directory."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    supported_extensions = {".csv", ".parquet", ".pq"}
    files = [p for p in raw_dir.iterdir() if p.suffix.lower() in supported_extensions]

    if not files:
        print(f"No matching files found in {raw_dir}")
        return

    for file_path in files:
        ext = file_path.suffix.lower()
        out_filename = f"{prefix}{file_path.stem}{ext}"
        out_path = sample_dir / out_filename

        try:
            if ext == ".csv":
                sample_csv(file_path, out_path, n_rows=sample_size)
            elif ext in {".parquet", ".pq"}:
                sample_parquet(file_path, out_path, n_rows=sample_size)

            raw_size_mb = file_path.stat().st_size / (1024 * 1024)
            sample_size_kb = out_path.stat().st_size / 1024
            print(
                f"  Original: {raw_size_mb:.2f} MB | "
                f"Sample: {sample_size_kb:.2f} KB ({sample_size} rows)"
            )
        except Exception as error:
            print(f"  Failed to process {file_path.name}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract compact samples from large data files for Git tracking."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the full datasets.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/samples"),
        help="Directory to store the sampled files.",
    )
    parser.add_argument(
        "--rows",
        "-n",
        type=int,
        default=1000,
        help="Number of rows per sample (default: 1000).",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default="sample_",
        help="Prefix prepended to sampled file names.",
    )

    args = parser.parse_args()
    process_directory(
        raw_dir=args.input_dir,
        sample_dir=args.output_dir,
        sample_size=args.rows,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()