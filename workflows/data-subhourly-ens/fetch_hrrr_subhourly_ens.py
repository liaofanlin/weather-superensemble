#!/usr/bin/env python3
"""
fetch_hrrr_subhourly_ens.py

Fetch selected HRRR 15-minute APCP messages from NOAA HRRR Open Data on AWS.
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


def read_inventory(date, cycle, aws_fhr):
    url = grib_url(date, cycle, aws_fhr) + ".idx"
    r = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "HRRR-subhourly-ensemble-workflow/1.0"},
    )
    r.raise_for_status()

    records = []
    for line in r.text.splitlines():
        if not line.strip():
            continue
        parts = line.split(":")
        if len(parts) < 5:
            continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        records.append({"line": line, "offset": offset})

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
        return {
            "start": rec["offset"],
            "end": end_offset,
        }

    available = [
        r["line"] for r in records if ":APCP:surface:" in r["line"]
    ]
    raise RuntimeError(
        f"Could not find 15-minute APCP record for "
        f"{start_min}-{end_min} min.\n"
        "Available APCP lines:\n  "
        + "\n  ".join(available)
    )


def valid_grib_message(path):
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        with path.open("rb") as f:
            if f.read(4) != b"GRIB":
                return False
            f.seek(-4, 2)
            return f.read(4) == b"7777"
    except OSError:
        return False


def download_record(date, cycle, aws_fhr, record, output_file):
    headers = {"User-Agent": "HRRR-subhourly-ensemble-workflow/1.0"}
    if record["end"] is None:
        headers["Range"] = f"bytes={record['start']}-"
    else:
        headers["Range"] = f"bytes={record['start']}-{record['end']}"

    url = grib_url(date, cycle, aws_fhr)
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()

    part = output_file.with_suffix(output_file.suffix + ".part")
    part.write_bytes(r.content)

    if not valid_grib_message(part):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Invalid GRIB2 message downloaded from {url}")

    part.replace(output_file)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--cycle", type=int, required=True)
    p.add_argument("--minute-start", type=int, required=True)
    p.add_argument("--minute-end", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.minute_start < 15 or args.minute_start % 15 != 0:
        raise SystemExit("ERROR: --minute-start must be 15, 30, 45, ...")
    if args.minute_end < args.minute_start or args.minute_end % 15 != 0:
        raise SystemExit("ERROR: invalid --minute-end")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventories = {}

    for valid_minute in range(args.minute_start, args.minute_end + 1, 15):
        aws_fhr = (valid_minute - 1) // 60 + 1

        output_file = output_dir / (
            f"hrrr.t{args.cycle:02d}z."
            f"wrfsubhf{aws_fhr:02d}."
            f"m{valid_minute:03d}.boulder.grib2"
        )

        if output_file.exists() and not args.overwrite:
            if valid_grib_message(output_file):
                print(f"EXISTS +{valid_minute:03d}: {output_file.name}")
                continue
            output_file.unlink()

        if aws_fhr not in inventories:
            inventories[aws_fhr] = read_inventory(
                args.date, args.cycle, aws_fhr
            )

        record = find_15min_apcp(inventories[aws_fhr], valid_minute)
        print(f"FETCH +{valid_minute:03d} from wrfsubhf{aws_fhr:02d}")
        download_record(
            args.date,
            args.cycle,
            aws_fhr,
            record,
            output_file,
        )


if __name__ == "__main__":
    main()
