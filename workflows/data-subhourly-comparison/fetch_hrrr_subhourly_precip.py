#!/usr/bin/env python3
"""
fetch_hrrr_subhourly_precip.py

Fetch HRRR 15-minute APCP messages from the NOAA HRRR Open Data AWS bucket.

Example:
    --fhr-start 0 --fhr-end 3

means forecast valid times from +00:15 through +03:00 every 15 minutes.

AWS file mapping:
    forecast block 0-1 h -> wrfsubhf01.grib2
    forecast block 1-2 h -> wrfsubhf02.grib2
    forecast block 2-3 h -> wrfsubhf03.grib2

For each wrfsubhf file, this script:
  1. downloads the .idx inventory,
  2. locates the four 15-minute APCP messages,
  3. uses HTTP Range requests to fetch only those GRIB2 messages.

No AWS account, AWS CLI, boto3, or s3fs is required.
"""

import argparse
import re
from pathlib import Path

import requests


AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"


def grib_url(date, cycle, aws_fhr):
    return (
        f"{AWS_BASE}/hrrr.{date}/conus/"
        f"hrrr.t{cycle:02d}z.wrfsubhf{aws_fhr:02d}.grib2"
    )


def idx_url(date, cycle, aws_fhr):
    return grib_url(date, cycle, aws_fhr) + ".idx"


def read_inventory(date, cycle, aws_fhr):
    url = idx_url(date, cycle, aws_fhr)

    r = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "HRRR-Denver-subhourly-AWS-workflow/1.0"},
    )
    r.raise_for_status()

    lines = [line for line in r.text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Empty HRRR inventory: {url}")

    records = []

    for line in lines:
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


def accumulation_window_minutes(line):
    s = line.lower()

    m = re.search(
        r"(?:^|:|\s)(\d+)\s*-\s*(\d+)\s+min(?:ute)?s?\s+acc\s+fcst",
        s,
    )
    if m:
        return int(m.group(1)), int(m.group(2))

    m = re.search(
        r"(?:^|:|\s)(\d+)\s*-\s*(\d+)\s+hour(?:s)?\s+acc\s+fcst",
        s,
    )
    if m:
        return 60 * int(m.group(1)), 60 * int(m.group(2))

    return None


def find_15min_apcp(records, valid_minute):
    start_min = valid_minute - 15
    end_min = valid_minute
    matches = []

    for i, rec in enumerate(records):
        line = rec["line"]

        if ":APCP:surface:" not in line:
            continue

        if accumulation_window_minutes(line) != (start_min, end_min):
            continue

        end_offset = (
            records[i + 1]["offset"] - 1
            if i + 1 < len(records)
            else None
        )

        matches.append({
            "line": line,
            "start": rec["offset"],
            "end": end_offset,
        })

    if not matches:
        available = [
            r["line"]
            for r in records
            if ":APCP:surface:" in r["line"]
        ]

        raise RuntimeError(
            f"Could not find 15-minute APCP record for "
            f"{start_min}-{end_min} min.\n"
            "Available APCP inventory lines:\n  "
            + "\n  ".join(available)
        )

    if len(matches) > 1:
        print(
            f"WARNING: found {len(matches)} APCP matches for "
            f"{start_min}-{end_min} min; using the first."
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


def download_record(date, cycle, aws_fhr, record, output_file):
    headers = {
        "User-Agent": "HRRR-Denver-subhourly-AWS-workflow/1.0",
    }

    if record["end"] is None:
        headers["Range"] = f"bytes={record['start']}-"
    else:
        headers["Range"] = f"bytes={record['start']}-{record['end']}"

    url = grib_url(date, cycle, aws_fhr)

    r = requests.get(
        url,
        headers=headers,
        timeout=120,
    )
    r.raise_for_status()

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
            "Downloaded APCP message is not a valid GRIB2 message "
            f"(size={size} bytes)."
        )

    part.replace(output_file)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYYMMDD")
    p.add_argument("--cycle", type=int, required=True, help="0-23 UTC")
    p.add_argument("--fhr-start", type=int, default=0)
    p.add_argument("--fhr-end", type=int, default=3)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.fhr_start < 0:
        raise SystemExit("ERROR: --fhr-start must be >= 0.")

    if args.fhr_end <= args.fhr_start:
        raise SystemExit(
            "ERROR: --fhr-end must be greater than --fhr-start."
        )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    first_valid_minute = args.fhr_start * 60 + 15
    last_valid_minute = args.fhr_end * 60

    print("=" * 72)
    print("Fetch HRRR 15-minute precipitation from AWS")
    print("=" * 72)
    print(f"Cycle          : {args.date} {args.cycle:02d}Z")
    print(
        f"Forecast range : +{first_valid_minute:03d} to "
        f"+{last_valid_minute:03d} min"
    )
    print("Output cadence : 15 minutes")
    print(
        f"AWS products   : wrfsubhf{args.fhr_start + 1:02d} "
        f"through wrfsubhf{args.fhr_end:02d}"
    )
    print("Source         : noaa-hrrr-bdp-pds")
    print("Method         : .idx inventory + HTTP byte-range")
    print(f"Output dir     : {outdir}")
    print()

    for block_hour in range(args.fhr_start, args.fhr_end):
        aws_fhr = block_hour + 1
        block_start = block_hour * 60

        block_valid_minutes = (
            block_start + 15,
            block_start + 30,
            block_start + 45,
            block_start + 60,
        )

        print("-" * 72)
        print(
            f"Forecast block {block_hour}-{block_hour + 1} h "
            f"-> wrfsubhf{aws_fhr:02d}"
        )
        print("-" * 72)

        records = read_inventory(
            args.date,
            args.cycle,
            aws_fhr,
        )

        for valid_minute in block_valid_minutes:
            start_min = valid_minute - 15

            output_file = (
                outdir
                / (
                    f"hrrr.t{args.cycle:02d}z."
                    f"wrfsubhf{aws_fhr:02d}."
                    f"m{valid_minute:03d}.denver.grib2"
                )
            )

            print(f"+{valid_minute:03d} min: ", end="")

            if output_file.exists() and not args.overwrite:
                if valid_grib_message(output_file):
                    print("exists and valid, skipping")
                    continue

                print("existing file invalid; replacing")
                output_file.unlink()

            record = find_15min_apcp(
                records,
                valid_minute,
            )

            print(
                f"fetching {start_min:03d}-{valid_minute:03d} "
                "min APCP..."
            )
            print(f"      inventory: {record['line']}")

            download_record(
                args.date,
                args.cycle,
                aws_fhr,
                record,
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
