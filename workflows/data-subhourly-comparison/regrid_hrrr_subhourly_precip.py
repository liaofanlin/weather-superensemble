#!/usr/bin/env python3
"""
regrid_hrrr_subhourly_precip.py

Read AWS-fetched HRRR 15-minute APCP GRIB messages, crop the native HRRR
grid around the requested Denver domain, interpolate to a regular 0.01-degree
grid, and save all requested valid times in one NetCDF file.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import griddata


SOURCE_MARGIN_DEG = 0.10


def open_subhourly_precip(grib_file):
    # The fetch script saves exactly one APCP GRIB message per file.
    ds = xr.open_dataset(
        grib_file,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )

    if "tp" not in ds:
        raise KeyError(f"'tp' not found in {grib_file}")

    return ds


def make_target_grid(lat_min, lat_max, lon_min, lon_max, resolution):
    lats = np.linspace(
        lat_min,
        lat_max,
        int(round((lat_max - lat_min) / resolution)) + 1,
    )
    lons = np.linspace(
        lon_min,
        lon_max,
        int(round((lon_max - lon_min) / resolution)) + 1,
    )

    lon2d, lat2d = np.meshgrid(lons, lats)
    return lats, lons, lat2d, lon2d


def crop_native_grid(lats, lons, values, lat_min, lat_max, lon_min, lon_max):
    lons = np.where(lons > 180.0, lons - 360.0, lons)

    mask = (
        (lats >= lat_min - SOURCE_MARGIN_DEG)
        & (lats <= lat_max + SOURCE_MARGIN_DEG)
        & (lons >= lon_min - SOURCE_MARGIN_DEG)
        & (lons <= lon_max + SOURCE_MARGIN_DEG)
    )

    iy, ix = np.where(mask)
    if len(iy) == 0:
        raise RuntimeError("No HRRR grid points intersect requested domain.")

    y0, y1 = iy.min(), iy.max() + 1
    x0, x1 = ix.min(), ix.max() + 1

    return (
        lats[y0:y1, x0:x1],
        lons[y0:y1, x0:x1],
        values[y0:y1, x0:x1],
    )


def regrid_field(src_lats, src_lons, src_values, target_lat2d, target_lon2d):
    good = np.isfinite(src_lats) & np.isfinite(src_lons) & np.isfinite(src_values)

    points = np.column_stack((src_lons[good], src_lats[good]))
    vals = src_values[good]

    out = griddata(
        points,
        vals,
        (target_lon2d, target_lat2d),
        method="linear",
    )

    missing = ~np.isfinite(out)
    if np.any(missing):
        nearest = griddata(
            points,
            vals,
            (target_lon2d, target_lat2d),
            method="nearest",
        )
        out[missing] = nearest[missing]

    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--cycle", type=int, required=True)
    p.add_argument("--fhr-start", type=int, default=0)
    p.add_argument("--fhr-end", type=int, default=3)
    p.add_argument("--lat-min", type=float, required=True)
    p.add_argument("--lat-max", type=float, required=True)
    p.add_argument("--lon-min", type=float, required=True)
    p.add_argument("--lon-max", type=float, required=True)
    p.add_argument("--resolution", type=float, default=0.01)
    p.add_argument("--output-file", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    if args.fhr_start < 0:
        raise SystemExit("ERROR: --fhr-start must be >= 0.")
    if args.fhr_end <= args.fhr_start:
        raise SystemExit(
            "ERROR: --fhr-end must be greater than --fhr-start."
        )

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    target_lats, target_lons, target_lat2d, target_lon2d = make_target_grid(
        args.lat_min,
        args.lat_max,
        args.lon_min,
        args.lon_max,
        args.resolution,
    )

    print("=" * 72)
    print("Regrid HRRR 15-minute precipitation")
    print("=" * 72)
    print(
        f"Target domain: {args.lat_min:.4f}-{args.lat_max:.4f} N, "
        f"{args.lon_min:.4f}-{args.lon_max:.4f} E"
    )
    print(
        f"Target grid  : {len(target_lats)} x {len(target_lons)} "
        f"({args.resolution:.3f} degree)"
    )
    valid_minutes = np.arange(
        args.fhr_start * 60 + 15,
        args.fhr_end * 60 + 1,
        15,
        dtype=np.int32,
    )

    print(
        f"Forecast range: +{valid_minutes[0]:03d} through "
        f"+{valid_minutes[-1]:03d} min"
    )
    print(f"Time steps    : {len(valid_minutes)} (15-minute cadence)")
    print()

    fields = []
    valid_times = []
    init_time = None

    for valid_minute in valid_minutes:
        valid_minute = int(valid_minute)
        aws_fhr = (valid_minute - 1) // 60 + 1

        grib_file = (
            input_dir
            / (
                f"hrrr.t{args.cycle:02d}z."
                f"wrfsubhf{aws_fhr:02d}."
                f"m{valid_minute:03d}.denver.grib2"
            )
        )

        print(
            f"+{valid_minute:03d} min "
            f"(wrfsubhf{aws_fhr:02d}): {grib_file.name}"
        )

        if not grib_file.exists():
            raise FileNotFoundError(grib_file)

        ds = open_subhourly_precip(grib_file)

        try:
            precip = ds["tp"]
            values = np.squeeze(precip.values.astype(float))

            units = str(precip.attrs.get("units", "")).lower()
            if units in {"m", "metre", "meter"}:
                values *= 1000.0

            values = np.maximum(values, 0.0)

            src_lats = np.asarray(ds["latitude"].values)
            src_lons = np.asarray(ds["longitude"].values)
            src_lons = np.where(src_lons > 180.0, src_lons - 360.0, src_lons)

            src_lats, src_lons, values = crop_native_grid(
                src_lats,
                src_lons,
                values,
                args.lat_min,
                args.lat_max,
                args.lon_min,
                args.lon_max,
            )

            out = regrid_field(
                src_lats,
                src_lons,
                values,
                target_lat2d,
                target_lon2d,
            )

            print(f"    cropped native grid : {values.shape}")
            print(f"    regridded grid      : {out.shape}")

            fields.append(out)

            if "valid_time" in ds.coords:
                valid_times.append(
                    np.asarray(ds["valid_time"].values).astype("datetime64[ns]")
                )
            else:
                valid_times.append(np.datetime64("NaT"))

            if init_time is None and "time" in ds.coords:
                init_time = np.asarray(ds["time"].values).astype("datetime64[ns]")

        finally:
            ds.close()

    data = np.stack(fields, axis=0)

    out_ds = xr.Dataset(
        data_vars={
            "precip_15min_mm": (
                ("forecast_minute", "latitude", "longitude"),
                data,
                {
                    "long_name": "HRRR 15-minute accumulated precipitation",
                    "units": "mm",
                    "source": "NOAA HRRR Open Data on AWS",
                    "regridding": (
                        "linear interpolation from native HRRR grid; "
                        "nearest-neighbor edge fill"
                    ),
                },
            )
        },
        coords={
            "forecast_minute": valid_minutes,
            "latitude": target_lats,
            "longitude": target_lons,
            "valid_time": (
                "forecast_minute",
                np.asarray(valid_times, dtype="datetime64[ns]"),
            ),
        },
        attrs={
            "model": "HRRR",
            "product": (
                f"wrfsubhf{args.fhr_start + 1:02d}-"
                f"wrfsubhf{args.fhr_end:02d}"
            ),
            "forecast_hour_start": args.fhr_start,
            "forecast_hour_end": args.fhr_end,
            "output_cadence_minutes": 15,
            "aws_bucket": "noaa-hrrr-bdp-pds",
            "native_horizontal_resolution": "approximately 3 km",
            "target_grid_resolution_degrees": args.resolution,
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
        },
    )

    if init_time is not None:
        out_ds.attrs["initialization_time"] = str(init_time)

    out_ds.to_netcdf(output_file)

    print()
    print(f"Saved: {output_file}")
    print(f"Dimensions: {dict(out_ds.sizes)}")


if __name__ == "__main__":
    main()
