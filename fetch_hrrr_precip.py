#!/usr/bin/env python3
"""
fetch_hrrr_precip.py

Fetch only the HRRR hourly APCP GRIB2 message from the NOAA HRRR
Open Data AWS bucket.

The AWS bucket contains full HRRR GRIB2 files plus .idx inventory files.
This script:
  1. downloads the .idx text file,
  2. locates the previous-hour APCP message,
  3. uses an HTTP Range request to fetch only that GRIB2 message.

Therefore it does NOT download the entire HRRR surface file.

No AWS account, AWS CLI, boto3, or s3fs is required.
"""

import argparse
from pathlib import Path

import requests


AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"


def grib_url(date, cycle, fhr):
    return (
        f"{AWS_BASE}/hrrr.{date}/conus/"
        f"hrrr.t{cycle:02d}z.wrfsfcf{fhr:02d}.grib2"
    )


def idx_url(date, cycle, fhr):
    return grib_url(date, cycle, fhr) + ".idx"


def read_inventory(date, cycle, fhr):
    url = idx_url(date, cycle, fhr)

    r = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "HRRR-Denver-AWS-workflow/1.0"},
    )
    r.raise_for_status()

    lines = [line for line in r.text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Empty HRRR inventory: {url}")

    records = []

    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 5:
            continue

        try:
            offset = int(parts[1])
        except ValueError:
            continue

        records.append({
            "line": line,
            "offset": offset,
        })

    if not records:
        raise RuntimeError(f"Could not parse HRRR inventory: {url}")

    return records


def find_hourly_apcp(records, fhr):
    # HRRR inventory normally contains both run-total APCP and the previous-hour
    # accumulation. We want:
    #   F01 -> 0-1 hour acc fcst
    #   F02 -> 1-2 hour acc fcst
    #   ...
    wanted = f"{fhr - 1}-{fhr} hour acc fcst"

    matches = []

    for i, rec in enumerate(records):
        line = rec["line"]

        if ":APCP:surface:" not in line:
            continue

        if wanted not in line:
            continue

        end = (
            records[i + 1]["offset"] - 1
            if i + 1 < len(records)
            else None
        )

        matches.append({
            "line": line,
            "start": rec["offset"],
            "end": end,
        })

    if not matches:
        # Helpful diagnostic: show all APCP records present.
        available = [
            r["line"]
            for r in records
            if ":APCP:surface:" in r["line"]
        ]

        raise RuntimeError(
            f"Could not find APCP record '{wanted}'.\n"
            "Available APCP inventory lines:\n  "
            + "\n  ".join(available)
        )

    if len(matches) > 1:
        print(
            f"WARNING: found {len(matches)} matches for {wanted}; "
            "using the first."
        )

    return matches[0]


def valid_grib_message(path):
    if not path.exists() or path.stat().st_size < 100:
        return False

    try:
        with path.open("rb") as f:
            start = f.read(4)
            f.seek(-4, 2)
            end = f.read(4)
        return start == b"GRIB" and end == b"7777"
    except OSError:
        return False


def download_message(date, cycle, fhr, output_file):
    records = read_inventory(date, cycle, fhr)
    record = find_hourly_apcp(records, fhr)

    print(f"      inventory: {record['line']}")

    headers = {
        "User-Agent": "HRRR-Denver-AWS-workflow/1.0",
    }

    if record["end"] is None:
        range_value = f"bytes={record['start']}-"
    else:
        range_value = f"bytes={record['start']}-{record['end']}"

    headers["Range"] = range_value

    url = grib_url(date, cycle, fhr)

    r = requests.get(
        url,
        headers=headers,
        timeout=120,
    )
    r.raise_for_status()

    # Range request should normally return 206 Partial Content.
    if r.status_code not in (200, 206):
        raise RuntimeError(
            f"Unexpected HTTP status {r.status_code} for {url}"
        )

    part = output_file.with_suffix(output_file.suffix + ".part")
    part.write_bytes(r.content)

    if not valid_grib_message(part):
        size = part.stat().st_size if part.exists() else 0
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded APCP message is not a valid GRIB2 message "
            f"(size={size} bytes)."
        )

    part.replace(output_file)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--cycle", type=int, required=True)
    p.add_argument("--fhr-start", type=int, default=1)
    p.add_argument("--fhr-end", type=int, default=6)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Fetch HRRR hourly precipitation from AWS")
    print("=" * 72)
    print(f"Cycle       : {args.date} {args.cycle:02d}Z")
    print(f"Forecast hrs: F{args.fhr_start:02d}-F{args.fhr_end:02d}")
    print("Source      : noaa-hrrr-bdp-pds")
    print("Method      : .idx inventory + HTTP byte-range")
    print(f"Output dir  : {outdir}")
    print()

    for fhr in range(args.fhr_start, args.fhr_end + 1):
        output_file = (
            outdir /
            f"hrrr.t{args.cycle:02d}z.wrfsfcf{fhr:02d}.denver.grib2"
        )

        print(f"F{fhr:02d}: ", end="")

        if output_file.exists() and not args.overwrite:
            if valid_grib_message(output_file):
                print("exists and valid, skipping")
                continue
            print("existing file invalid; replacing")
            output_file.unlink()

        print("fetching APCP message...")
        download_message(
            args.date,
            args.cycle,
            fhr,
            output_file,
        )

        print(
            f"      saved: {output_file.name} "
            f"({output_file.stat().st_size / 1024 / 1024:.2f} MB)"
        )

    print()
    print("Fetch complete.")


if __name__ == "__main__":
    main()
