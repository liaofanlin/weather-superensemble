#!/usr/bin/env python3
"""
plot_hrrr_subhourly_ens.py

Creates one 2x2 panel figure for each common valid time.
"""

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import requests
import xarray as xr


TIGER_COUNTY_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
    "TIGERweb/State_County/MapServer/1/query"
)

LEVELS = np.array([
    0.0508,
    0.2540,
    1.2700,
    2.5400,
    6.3500,
    12.7000,
    19.0500,
    25.4000,
    50.8000,
])

PRECIP_COLORS = [
    "#808080",
    "#00DCE6",
    "#5EE600",
    "#B7FF3C",
    "#FFE600",
    "#FFB515",
    "#FF6A00",
    "#FF310D",
]


def precip_style():
    cmap = ListedColormap(PRECIP_COLORS)
    cmap.set_under("white")
    cmap.set_over("#D96AD9")
    norm = BoundaryNorm(LEVELS, ncolors=cmap.N, clip=False)
    return cmap, norm


def fetch_county_boundaries(bbox):
    south, west, north, east = bbox
    params = {
        "where": "1=1",
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "NAME,STATE,COUNTY",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }

    r = requests.get(
        TIGER_COUNTY_URL,
        params=params,
        timeout=60,
        headers={"User-Agent": "HRRR-Boulder-subhourly-ensemble/1.0"},
    )
    r.raise_for_status()

    counties = []
    for feature in r.json().get("features", []):
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates")
        geom_type = geom.get("type")

        if not coords:
            continue
        if geom_type == "Polygon":
            polygons = [coords]
        elif geom_type == "MultiPolygon":
            polygons = coords
        else:
            continue

        counties.append({"polygons": polygons})

    return counties


def draw_county_boundaries(ax, counties, transform=None):
    for county in counties:
        for polygon in county["polygons"]:
            for ring in polygon:
                if len(ring) < 2:
                    continue
                lons = [p[0] for p in ring]
                lats = [p[1] for p in ring]

                kwargs = {
                    "color": "black",
                    "linewidth": 0.75,
                    "alpha": 0.9,
                    "zorder": 20,
                }
                if transform is not None:
                    kwargs["transform"] = transform

                ax.plot(lons, lats, **kwargs)


def grid_cell_extent(coords):
    coords = np.asarray(coords, dtype=float)
    spacing = np.median(np.diff(coords))
    return (
        float(coords[0] - 0.5 * spacing),
        float(coords[-1] + 0.5 * spacing),
    )


def parse_valid_time(value):
    s = np.datetime_as_string(
        np.asarray(value).astype("datetime64[m]"),
        unit="m",
    )
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def lead_string(minutes):
    return f"+{minutes // 60:02d}:{minutes % 60:02d}"


def plot_panel(base_init, valid_offset_min, member_files, output_file, no_counties):
    datasets = [xr.open_dataset(path) for path in member_files]

    try:
        lats = np.asarray(datasets[0]["latitude"].values)
        lons = np.asarray(datasets[0]["longitude"].values)

        lon_left, lon_right = grid_cell_extent(lons)
        lat_bottom, lat_top = grid_cell_extent(lats)

        bbox = (
            float(lats.min()),
            float(lons.min()),
            float(lats.max()),
            float(lons.max()),
        )
        counties = [] if no_counties else fetch_county_boundaries(bbox)

        cmap, norm = precip_style()

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            projection = ccrs.PlateCarree()
            use_cartopy = True
        except ImportError:
            projection = None
            cfeature = None
            use_cartopy = False

        if use_cartopy:
            fig, axes = plt.subplots(
                2,
                2,
                figsize=(11, 10),
                subplot_kw={"projection": projection},
            )
        else:
            fig, axes = plt.subplots(2, 2, figsize=(11, 10))

        axes = axes.ravel()
        pcm = None
        common_valid = None

        for member_index, (ax, ds) in enumerate(zip(axes, datasets)):
            lead_min = valid_offset_min + member_index * 60
            field = ds["precip_15min_mm"].sel(forecast_minute=lead_min)

            values = np.asarray(field.values)

            member_valid = parse_valid_time(
                ds["valid_time"].sel(forecast_minute=lead_min).values
            )

            if common_valid is None:
                common_valid = member_valid
            elif member_valid != common_valid:
                raise RuntimeError(
                    f"Valid times do not match: {common_valid} vs {member_valid}"
                )

            init_dt = base_init - timedelta(hours=member_index)

            imshow_kwargs = dict(
                origin="lower",
                extent=[lon_left, lon_right, lat_bottom, lat_top],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                resample=False,
                aspect="auto",
                zorder=1,
            )

            if use_cartopy:
                imshow_kwargs["transform"] = projection

            pcm = ax.imshow(values, **imshow_kwargs)

            if use_cartopy:
                ax.add_feature(cfeature.STATES, linewidth=0.6, zorder=4)
                ax.set_extent(
                    [lon_left, lon_right, lat_bottom, lat_top],
                    crs=projection,
                )
                ax.set_aspect("auto")
                ax.set_box_aspect(1)
                draw_county_boundaries(ax, counties, transform=projection)

                gl = ax.gridlines(
                    draw_labels=True,
                    linewidth=0.35,
                    alpha=0.4,
                    zorder=3,
                )
                gl.top_labels = False
                gl.right_labels = False

                if member_index % 2 == 1:
                    gl.left_labels = False
                if member_index < 2:
                    gl.bottom_labels = False
            else:
                draw_county_boundaries(ax, counties)
                ax.set_xlim(lon_left, lon_right)
                ax.set_ylim(lat_bottom, lat_top)
                ax.set_box_aspect(1)
                ax.grid(True, linewidth=0.35, alpha=0.4, zorder=2)

            ax.set_title(
                f"Member {member_index + 1}: Init {init_dt:%m/%d %HZ}\n"
                f"Lead {lead_string(lead_min)}",
                fontsize=11,
                pad=6,
            )

        fig.suptitle(
            "HRRR 15-min Precipitation — 4-member Time-Lagged Ensemble\n"
            f"Valid {common_valid:%Y-%m-%d %H:%M UTC}",
            fontsize=15,
            y=0.975,
        )

        fig.subplots_adjust(
            left=0.07,
            right=0.89,
            bottom=0.08,
            top=0.90,
            wspace=0.10,
            hspace=0.18,
        )

        cax = fig.add_axes([0.91, 0.16, 0.025, 0.66])
        cbar = fig.colorbar(pcm, cax=cax, extend="both")
        cbar.set_ticks(LEVELS)
        cbar.set_ticklabels([
            "0.0508",
            "0.254",
            "1.27",
            "2.54",
            "6.35",
            "12.7",
            "19.05",
            "25.4",
            "50.8",
        ])
        cbar.set_label("15-min precipitation (mm)")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    finally:
        for ds in datasets:
            ds.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-date", required=True)
    p.add_argument("--base-cycle", type=int, required=True)
    p.add_argument("--member-files", nargs=4, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-counties", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    base_init = datetime.strptime(
        f"{args.base_date}{args.base_cycle:02d}",
        "%Y%m%d%H",
    ).replace(tzinfo=timezone.utc)

    member_files = [Path(x) for x in args.member_files]
    output_dir = Path(args.output_dir)

    for valid_offset_min in range(15, 181, 15):
        valid_dt = base_init + timedelta(minutes=valid_offset_min)

        output_file = output_dir / (
            f"subhourly_ens_valid_{valid_dt:%Y%m%d_%H%M}UTC.png"
        )

        print(f"PLOT valid {valid_dt:%Y-%m-%d %H:%M UTC}")

        plot_panel(
            base_init,
            valid_offset_min,
            member_files,
            output_file,
            args.no_counties,
        )


if __name__ == "__main__":
    main()
