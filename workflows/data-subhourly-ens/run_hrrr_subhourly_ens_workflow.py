#!/usr/bin/env python3
"""
run_hrrr_subhourly_ens_workflow.py

Four-member time-lagged HRRR subhourly ensemble.
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

BOULDER_LAT = 40.0150
BOULDER_LON = -105.2705

DOMAIN_SIZE_DEG = 0.5
TARGET_RESOLUTION_DEG = 0.01

N_MEMBERS = 4

VALID_START_MIN = 15
VALID_END_MIN = 180

START = datetime(2026, 8, 26, 23, tzinfo=timezone.utc)

OVERWRITE_FETCH = False
NO_COUNTIES = False

AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "data-subhourly-ens"
    / "raw"
    / "hrrr-subhourly"
)

REGRID_ROOT = (
    PROJECT_ROOT
    / "data"
    / "data-subhourly-ens"
    / "processed"
    / "hrrr-subhourly"
)

FIG_ROOT = (
    PROJECT_ROOT
    / "figures"
    / "data-subhourly-ens"
)


def aws_idx_url(init_dt, aws_fhr=1):
    date = init_dt.strftime("%Y%m%d")
    cycle = init_dt.hour

    return (
        f"{AWS_BASE}/hrrr.{date}/conus/"
        f"hrrr.t{cycle:02d}z.wrfsubhf{aws_fhr:02d}.grib2.idx"
    )


def cycle_available(init_dt):
    try:
        r = requests.get(aws_idx_url(init_dt, 1), timeout=15)
        return r.status_code == 200 and ":APCP:surface:" in r.text
    except requests.RequestException:
        return False


def run_command(cmd):
    print()
    print("=" * 78)
    print(" ".join(str(x) for x in cmd))
    print("=" * 78)
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="Base cycle date YYYYMMDD")
    p.add_argument("--cycle", type=int, default=None, help="Base cycle UTC")
    return p.parse_args()


def resolve_start(args):
    if (args.date is None) != (args.cycle is None):
        raise SystemExit("ERROR: --date and --cycle must be supplied together.")

    if args.date is None:
        return START

    return datetime.strptime(
        f"{args.date}{args.cycle:02d}",
        "%Y%m%d%H",
    ).replace(tzinfo=timezone.utc)


def main():
    args = parse_args()
    base_init = resolve_start(args)

    half = DOMAIN_SIZE_DEG / 2.0
    lat_min = BOULDER_LAT - half
    lat_max = BOULDER_LAT + half
    lon_min = BOULDER_LON - half
    lon_max = BOULDER_LON + half

    member_inits = [
        base_init - timedelta(hours=i)
        for i in range(N_MEMBERS)
    ]

    print("=" * 78)
    print("HRRR BOULDER SUBHOURLY 4-MEMBER TIME-LAGGED ENSEMBLE")
    print("=" * 78)
    print(f"Base initialization : {base_init:%Y-%m-%d %H:%M UTC}")
    print("Valid times         : +00:15 through +03:00 every 15 min")
    print("Panel layout        : 2 x 2")
    print()

    member_nc_files = []

    for member_index, init_dt in enumerate(member_inits):
        member_number = member_index + 1
        date = init_dt.strftime("%Y%m%d")
        cycle = init_dt.hour

        minute_start = VALID_START_MIN + member_index * 60
        minute_end = VALID_END_MIN + member_index * 60

        print(
            f"M{member_number}: init {init_dt:%Y-%m-%d %HZ}, "
            f"lead +{minute_start:03d} to +{minute_end:03d} min"
        )

        if not cycle_available(init_dt):
            raise RuntimeError(
                f"HRRR cycle unavailable: {init_dt:%Y-%m-%d %HZ}"
            )

        member_tag = f"member{member_number}_{date}_{cycle:02d}z"

        raw_dir = RAW_ROOT / member_tag
        nc_file = (
            REGRID_ROOT
            / member_tag
            / "hrrr_precip_15min_001deg.nc"
        )
        member_nc_files.append(nc_file)

        fetch_cmd = [
            sys.executable,
            str(HERE / "fetch_hrrr_subhourly_ens.py"),
            "--date", date,
            "--cycle", str(cycle),
            "--minute-start", str(minute_start),
            "--minute-end", str(minute_end),
            "--output-dir", str(raw_dir),
        ]

        if OVERWRITE_FETCH:
            fetch_cmd.append("--overwrite")

        run_command(fetch_cmd)

        regrid_cmd = [
            sys.executable,
            str(HERE / "regrid_hrrr_subhourly_ens.py"),
            "--input-dir", str(raw_dir),
            "--date", date,
            "--cycle", str(cycle),
            "--minute-start", str(minute_start),
            "--minute-end", str(minute_end),
            "--lat-min", str(lat_min),
            "--lat-max", str(lat_max),
            "--lon-min", str(lon_min),
            "--lon-max", str(lon_max),
            "--resolution", str(TARGET_RESOLUTION_DEG),
            "--output-file", str(nc_file),
        ]

        run_command(regrid_cmd)

    fig_dir = FIG_ROOT / f"{base_init:%Y%m%d_%H}z"

    plot_cmd = [
        sys.executable,
        str(HERE / "plot_hrrr_subhourly_ens.py"),
        "--base-date", base_init.strftime("%Y%m%d"),
        "--base-cycle", str(base_init.hour),
        "--member-files",
        *[str(x) for x in member_nc_files],
        "--output-dir", str(fig_dir),
    ]

    if NO_COUNTIES:
        plot_cmd.append("--no-counties")

    run_command(plot_cmd)

    print()
    print("=" * 78)
    print("WORKFLOW COMPLETE")
    print("=" * 78)
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
