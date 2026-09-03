#!/usr/bin/env python3
"""
run_hrrr_subhourly_workflow.py

Master HRRR subhourly workflow using NOAA HRRR Open Data on AWS:

    AWS HRRR wrfsubhf01
      -> extract 15-min APCP valid at +15/+30/+45/+60 min
      -> crop Denver source region
      -> regrid to 0.01-degree lat/lon
      -> save NetCDF
      -> plot precipitation + major roads

Normally just run:
    python run_hrrr_subhourly_workflow.py

Or select a cycle:
    python run_hrrr_subhourly_workflow.py --date 20260903 --cycle 18
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests


# ============================================================
# USER SETTINGS
# ============================================================

DENVER_LAT = 39.7392
DENVER_LON = -104.9903

DOMAIN_SIZE_DEG = 0.5
TARGET_RESOLUTION_DEG = 0.01

FHR_START = 0
FHR_END = 3

# If both are None, the newest available AWS HRRR subhourly cycle is found.
DATE = None
CYCLE = None

REFRESH_ROADS = False
OVERWRITE_FETCH = False

AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

RAW_ROOT = PROJECT_ROOT / "data" / "data-ens-comparison" / "raw" / "hrrr-subhourly"
REGRID_ROOT = PROJECT_ROOT / "data" / "data-ens-comparison" / "processed" / "hrrr-subhourly"
FIG_ROOT = PROJECT_ROOT / "figures" / "data-ens-comparison" / "hrrr-subhourly"


def aws_idx_url(date, cycle, aws_fhr=1):
    return (
        f"{AWS_BASE}/hrrr.{date}/conus/"
        f"hrrr.t{cycle:02d}z.wrfsubhf{aws_fhr:02d}.grib2.idx"
    )


def cycle_available(date, cycle):
    url = aws_idx_url(date, cycle, 1)
    try:
        r = requests.get(url, timeout=15)
        return r.status_code == 200 and ":APCP:surface:" in r.text
    except requests.RequestException:
        return False


def find_latest_cycle(max_lookback_hours=24):
    # Publication can lag real time; begin three hours behind, matching the
    # behavior of the existing hourly workflow.
    #start = datetime.now(timezone.utc) - timedelta(hours=3)
    start = datetime(2026, 8, 26, 23, tzinfo=timezone.utc)

    print("Searching AWS for newest available HRRR subhourly cycle...")

    for back in range(max_lookback_hours + 1):
        dt = start - timedelta(hours=back)
        date = dt.strftime("%Y%m%d")
        cycle = dt.hour

        print(f"  checking {date} {cycle:02d}Z", end="")
        if cycle_available(date, cycle):
            print("  available")
            return date, cycle
        print("  not ready")

    raise RuntimeError(
        f"Could not find an available HRRR subhourly cycle within "
        f"{max_lookback_hours} hours."
    )


def resolve_cycle(date_arg, cycle_arg):
    if (date_arg is None) != (cycle_arg is None):
        raise SystemExit("ERROR: --date and --cycle must be supplied together.")

    if date_arg is not None:
        if not cycle_available(date_arg, cycle_arg):
            raise SystemExit(
                f"ERROR: AWS HRRR subhourly cycle {date_arg} {cycle_arg:02d}Z "
                "does not appear to be available."
            )
        return date_arg, cycle_arg

    if DATE is not None and CYCLE is not None:
        if not cycle_available(DATE, int(CYCLE)):
            raise SystemExit(
                f"ERROR: AWS HRRR subhourly cycle {DATE} {int(CYCLE):02d}Z "
                "does not appear to be available."
            )
        return DATE, int(CYCLE)

    return find_latest_cycle()


def run_command(cmd):
    print()
    print("=" * 78)
    print(" ".join(str(x) for x in cmd))
    print("=" * 78)
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYYMMDD")
    p.add_argument("--cycle", type=int, default=None, help="0-23 UTC")
    return p.parse_args()


def main():
    args = parse_args()

    if FHR_START < 0:
        raise SystemExit("ERROR: FHR_START must be >= 0.")
    if FHR_END <= FHR_START:
        raise SystemExit(
            "ERROR: FHR_END must be greater than FHR_START."
        )

    date, cycle = resolve_cycle(args.date, args.cycle)

    half = DOMAIN_SIZE_DEG / 2.0
    lat_min = DENVER_LAT - half
    lat_max = DENVER_LAT + half
    lon_min = DENVER_LON - half
    lon_max = DENVER_LON + half

    cycle_tag = f"{date}_{cycle:02d}z"

    raw_dir = RAW_ROOT / f"hrrr.{date}" / f"{cycle:02d}z"
    regrid_file = REGRID_ROOT / cycle_tag / "hrrr_precip_15min_001deg.nc"
    fig_dir = FIG_ROOT / cycle_tag

    print()
    print("=" * 78)
    print("HRRR DENVER SUBHOURLY AWS WORKFLOW")
    print("=" * 78)
    print(f"Cycle             : {date} {cycle:02d}Z")
    first_valid_minute = FHR_START * 60 + 15
    last_valid_minute = FHR_END * 60
    n_times = (last_valid_minute - first_valid_minute) // 15 + 1

    print(
        f"Source products   : wrfsubhf{FHR_START + 1:02d} "
        f"through wrfsubhf{FHR_END:02d}"
    )
    print("Source            : NOAA HRRR Open Data on AWS")
    print(
        f"Forecast valid    : +{first_valid_minute:03d} to "
        f"+{last_valid_minute:03d} min"
    )
    print(f"Output cadence    : 15 minutes ({n_times} times)")
    print(f"Denver center     : {DENVER_LAT:.4f}, {DENVER_LON:.4f}")
    print(f"Domain size       : {DOMAIN_SIZE_DEG:.2f} x {DOMAIN_SIZE_DEG:.2f} degree")
    print(f"Latitude          : {lat_min:.4f} to {lat_max:.4f}")
    print(f"Longitude         : {lon_min:.4f} to {lon_max:.4f}")
    print(f"Output resolution : {TARGET_RESOLUTION_DEG:.3f} degree")
    print("Output grid       : 51 x 51")

    fetch_cmd = [
        sys.executable,
        str(HERE / "fetch_hrrr_subhourly_precip.py"),
        "--date", date,
        "--cycle", str(cycle),
        "--fhr-start", str(FHR_START),
        "--fhr-end", str(FHR_END),
        "--output-dir", str(raw_dir),
    ]
    if OVERWRITE_FETCH:
        fetch_cmd.append("--overwrite")

    run_command(fetch_cmd)

    regrid_cmd = [
        sys.executable,
        str(HERE / "regrid_hrrr_subhourly_precip.py"),
        "--input-dir", str(raw_dir),
        "--cycle", str(cycle),
        "--fhr-start", str(FHR_START),
        "--fhr-end", str(FHR_END),
        "--lat-min", str(lat_min),
        "--lat-max", str(lat_max),
        "--lon-min", str(lon_min),
        "--lon-max", str(lon_max),
        "--resolution", str(TARGET_RESOLUTION_DEG),
        "--output-file", str(regrid_file),
    ]

    run_command(regrid_cmd)

    plot_cmd = [
        sys.executable,
        str(HERE / "plot_hrrr_subhourly_precip.py"),
        "--input-file", str(regrid_file),
        "--output-dir", str(fig_dir),
    ]
    if REFRESH_ROADS:
        plot_cmd.append("--refresh-roads")

    run_command(plot_cmd)

    print()
    print("=" * 78)
    print("WORKFLOW COMPLETE")
    print("=" * 78)
    print(f"15-min APCP GRIB  : {raw_dir}")
    print(f"Regridded NetCDF  : {regrid_file}")
    print(f"Figures            : {fig_dir}")


if __name__ == "__main__":
    main()
