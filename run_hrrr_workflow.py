#!/usr/bin/env python3
"""
run_hrrr_workflow.py

Master HRRR workflow using NOAA HRRR Open Data on AWS:

    AWS HRRR -> extract hourly APCP GRIB message
             -> crop Denver source region
             -> regrid to 0.01-degree lat/lon
             -> save NetCDF
             -> plot precipitation + major roads

Normally just run:
    python run_hrrr_workflow.py

Or select a cycle:
    python run_hrrr_workflow.py --date 20260831 --cycle 0
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

FHR_START = 1
FHR_END = 6

# If both are None, the newest available AWS HRRR cycle is found automatically.
DATE = None
CYCLE = None

REFRESH_ROADS = False
OVERWRITE_FETCH = False

AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

HERE = Path(__file__).resolve().parent
RAW_ROOT = HERE / "data" / "hrrr"
REGRID_ROOT = HERE / "data" / "regridded"
FIG_ROOT = HERE / "figures" / "hrrr_precip"


def aws_idx_url(date, cycle, fhr=1):
    return (
        f"{AWS_BASE}/hrrr.{date}/conus/"
        f"hrrr.t{cycle:02d}z.wrfsfcf{fhr:02d}.grib2.idx"
    )


def cycle_available(date, cycle):
    url = aws_idx_url(date, cycle, 1)
    try:
        r = requests.get(url, timeout=15)
        return r.status_code == 200 and "APCP" in r.text
    except requests.RequestException:
        return False


def find_latest_cycle(max_lookback_hours=24):
    # Start one hour behind real time because the newest cycle may still be publishing.
    start = datetime.now(timezone.utc) - timedelta(hours=10)

    print("Searching AWS for newest available HRRR cycle...")

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
        f"Could not find an available HRRR cycle within "
        f"{max_lookback_hours} hours."
    )


def resolve_cycle(date_arg, cycle_arg):
    if (date_arg is None) != (cycle_arg is None):
        raise SystemExit("ERROR: --date and --cycle must be supplied together.")

    if date_arg is not None:
        if not cycle_available(date_arg, cycle_arg):
            raise SystemExit(
                f"ERROR: AWS HRRR cycle {date_arg} {cycle_arg:02d}Z "
                "does not appear to be available."
            )
        return date_arg, cycle_arg

    if DATE is not None and CYCLE is not None:
        if not cycle_available(DATE, int(CYCLE)):
            raise SystemExit(
                f"ERROR: AWS HRRR cycle {DATE} {int(CYCLE):02d}Z "
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
    date, cycle = resolve_cycle(args.date, args.cycle)

    half = DOMAIN_SIZE_DEG / 2.0
    lat_min = DENVER_LAT - half
    lat_max = DENVER_LAT + half
    lon_min = DENVER_LON - half
    lon_max = DENVER_LON + half

    cycle_tag = f"{date}_{cycle:02d}z"

    raw_dir = RAW_ROOT / f"hrrr.{date}" / f"{cycle:02d}z"
    regrid_file = REGRID_ROOT / cycle_tag / "hrrr_precip_001deg.nc"
    fig_dir = FIG_ROOT / cycle_tag

    print()
    print("=" * 78)
    print("HRRR DENVER AWS WORKFLOW")
    print("=" * 78)
    print(f"Cycle             : {date} {cycle:02d}Z")
    print(f"Source            : NOAA HRRR Open Data on AWS")
    print(f"Denver center     : {DENVER_LAT:.4f}, {DENVER_LON:.4f}")
    print(f"Domain size       : {DOMAIN_SIZE_DEG:.2f} x {DOMAIN_SIZE_DEG:.2f} degree")
    print(f"Latitude          : {lat_min:.4f} to {lat_max:.4f}")
    print(f"Longitude         : {lon_min:.4f} to {lon_max:.4f}")
    print(f"Output resolution : {TARGET_RESOLUTION_DEG:.3f} degree")
    print(f"Output grid       : 51 x 51")
    print(f"Forecast hours    : F{FHR_START:02d}-F{FHR_END:02d}")

    fetch_cmd = [
        sys.executable,
        str(HERE / "fetch_hrrr_precip.py"),
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
        str(HERE / "regrid_hrrr_precip.py"),
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
        str(HERE / "plot_hrrr_precip.py"),
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
    print(f"Hourly APCP GRIB : {raw_dir}")
    print(f"Regridded NetCDF : {regrid_file}")
    print(f"Figures          : {fig_dir}")


if __name__ == "__main__":
    main()
