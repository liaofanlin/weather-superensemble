#!/usr/bin/env python3
"""
plot_hrrr_subhourly_precip.py

Plot regridded HRRR 15-minute precipitation on the 51 x 51 Denver grid.

Important plotting choices:
  * The precipitation field is rendered as a true raster with imshow().
  * interpolation="nearest" preserves the visible 51 x 51 grid cells.
  * The map panel is explicitly square.
  * Only selected major Denver highways are overlaid.

County boundary source:
    U.S. Census TIGERweb State_County service

County boundaries are drawn instead of highways.

No geopandas/osmnx required.
"""

import argparse
import re
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


def fetch_county_boundaries(bbox):
    """
    Download county polygons intersecting the plotting domain from TIGERweb.
    """
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

    print("Downloading county boundaries from TIGERweb...")

    r = requests.get(
        TIGER_COUNTY_URL,
        params=params,
        timeout=60,
        headers={"User-Agent": "HRRR-Denver-county-boundaries/1.0"},
    )
    r.raise_for_status()

    payload = r.json()
    counties = []

    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
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

        counties.append({
            "name": props.get("NAME", "County"),
            "polygons": polygons,
        })

    print(f"Counties loaded       : {len(counties)}")
    if counties:
        print(
            "County names          : "
            + ", ".join(sorted({c["name"] for c in counties}))
        )

    return counties


def draw_county_boundaries(ax, counties, transform=None):
    """
    Draw polygon rings as county boundary lines.
    """
    for county in counties:
        for polygon in county["polygons"]:
            for ring in polygon:
                if len(ring) < 2:
                    continue

                lons = [p[0] for p in ring]
                lats = [p[1] for p in ring]

                kwargs = {
                    "color": "black",
                    "linewidth": 0.9,
                    "alpha": 0.9,
                    "zorder": 20,
                }

                if transform is not None:
                    kwargs["transform"] = transform

                ax.plot(lons, lats, **kwargs)


def format_time(value):
    try:
        if np.isnat(value):
            return None
    except TypeError:
        pass

    return np.datetime_as_string(value, unit="m")


def grid_cell_extent(coords):
    """
    Return image edges so the first/last data values are cell centers.

    For a regular 1-D grid this extends the domain by half a grid spacing
    on each side, which is appropriate for a raster representation.
    """
    coords = np.asarray(coords, dtype=float)

    if coords.size == 1:
        return coords[0] - 0.5, coords[0] + 0.5

    spacing = np.median(np.diff(coords))

    return (
        float(coords[0] - 0.5 * spacing),
        float(coords[-1] + 0.5 * spacing),
    )


def format_forecast_offset(minutes):
    hours = int(minutes) // 60
    mins = int(minutes) % 60
    return f"+{hours:02d}:{mins:02d}"


def plot_one(ds, forecast_minute, output_file, counties):
    field = ds["precip_15min_mm"].sel(forecast_minute=forecast_minute)

    lats = np.asarray(ds["latitude"].values)
    lons = np.asarray(ds["longitude"].values)
    values = np.asarray(field.values)

    if values.shape != (len(lats), len(lons)):
        raise ValueError(
            f"Unexpected precipitation shape {values.shape}; "
            f"expected ({len(lats)}, {len(lons)})"
        )

    # Ensure increasing coordinate order for imshow.
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        values = values[::-1, :]

    if lons[0] > lons[-1]:
        lons = lons[::-1]
        values = values[:, ::-1]

    lon_left, lon_right = grid_cell_extent(lons)
    lat_bottom, lat_top = grid_cell_extent(lats)

    levels = np.array([
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

    precip_colors = [
        "#808080",
        "#00DCE6",
        "#5EE600",
        "#B7FF3C",
        "#FFE600",
        "#FFB515",
        "#FF6A00",
        "#FF310D",
    ]

    precip_cmap = ListedColormap(precip_colors)
    precip_cmap.set_under("white")
    precip_cmap.set_over("#D96AD9")

    precip_norm = BoundaryNorm(
        levels,
        ncolors=precip_cmap.N,
        clip=False,
    )

    title = (
        f"HRRR 15-min precipitation: "
        f"{format_forecast_offset(forecast_minute)}\n"
        f"Accumulation "
        f"{format_forecast_offset(forecast_minute - 15)} to "
        f"{format_forecast_offset(forecast_minute)}"
    )

    init_time = ds.attrs.get("initialization_time")

    if init_time:
        init_time = str(init_time).replace(".000000000", "")
        title += f" | Init {init_time}"

    if "valid_time" in ds:
        valid_time = format_time(
            ds["valid_time"].sel(forecast_minute=forecast_minute).values
        )

        if valid_time:
            title += f"\nValid {valid_time} UTC"

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        # Square figure, with an explicitly square map axes.
        fig = plt.figure(figsize=(9, 9))

        # [left, bottom, width, height] -- width == height => square panel.
        ax = fig.add_axes(
            [0.10, 0.12, 0.72, 0.72],
            projection=ccrs.PlateCarree(),
        )

        # TRUE RASTER:
        # Each of the 51 x 51 values is drawn as one rectangular image cell.
        # interpolation="nearest" prevents smoothing/interpolation.
        pcm = ax.imshow(
            values,
            origin="lower",
            extent=[
                lon_left,
                lon_right,
                lat_bottom,
                lat_top,
            ],
            cmap=precip_cmap,
            norm=precip_norm,
            interpolation="nearest",
            resample=False,
            transform=ccrs.PlateCarree(),
            zorder=1,
            aspect="auto",
        )

        ax.add_feature(
            cfeature.STATES,
            linewidth=0.7,
            zorder=4,
        )

        ax.set_extent(
            [
                lon_left,
                lon_right,
                lat_bottom,
                lat_top,
            ],
            crs=ccrs.PlateCarree(),
        )

        # Prevent Cartopy from reshaping the axes according to geographic
        # aspect ratio; we want the 51 x 51 model grid displayed as a square.
        ax.set_aspect("auto")
        ax.set_box_aspect(1)

        draw_county_boundaries(
            ax,
            counties,
            transform=ccrs.PlateCarree(),
        )

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.4,
            alpha=0.4,
            zorder=3,
        )
        gl.top_labels = False
        gl.right_labels = False

        # Separate colorbar axes so adding the colorbar cannot resize
        # the square map panel.
        cax = fig.add_axes([0.85, 0.12, 0.035, 0.72])

        cbar = fig.colorbar(
            pcm,
            cax=cax,
            extend="both",
        )

    except ImportError:
        fig = plt.figure(figsize=(9, 9))

        ax = fig.add_axes(
            [0.10, 0.12, 0.72, 0.72]
        )

        pcm = ax.imshow(
            values,
            origin="lower",
            extent=[
                lon_left,
                lon_right,
                lat_bottom,
                lat_top,
            ],
            cmap=precip_cmap,
            norm=precip_norm,
            interpolation="nearest",
            resample=False,
            zorder=1,
            aspect="auto",
        )

        draw_county_boundaries(ax, counties)

        ax.set_xlim(lon_left, lon_right)
        ax.set_ylim(lat_bottom, lat_top)

        ax.set_box_aspect(1)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.4,
            zorder=2,
        )

        cax = fig.add_axes([0.85, 0.12, 0.035, 0.72])

        cbar = fig.colorbar(
            pcm,
            cax=cax,
            extend="both",
        )

    cbar.set_ticks(levels)
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

    ax.set_title(
        title,
        pad=10,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Do not use tight_layout() or bbox_inches="tight" here:
    # the axes positions are intentional and keep the map panel square.
    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input-file",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    p.add_argument(
        "--no-counties",
        action="store_true",
        help="Do not draw county boundaries.",
    )

    # Backward-compatible alias from the highway version.
    p.add_argument(
        "--no-roads",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    # Accepted for compatibility with the master script.
    # There is intentionally no road cache in this version.
    p.add_argument(
        "--refresh-roads",
        action="store_true",
    )

    return p.parse_args()


def main():
    args = parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)

    ds = xr.open_dataset(input_file)

    try:
        lats = ds["latitude"].values
        lons = ds["longitude"].values

        bbox = (
            float(lats.min()),
            float(lons.min()),
            float(lats.max()),
            float(lons.max()),
        )

        print("=" * 72)
        print("Plot regridded HRRR 15-minute precipitation")
        print("=" * 72)

        print(f"Input file   : {input_file}")
        print(f"Grid         : {len(lats)} x {len(lons)}")

        print(
            f"Domain       : "
            f"{lats.min():.4f}-{lats.max():.4f} N, "
            f"{lons.min():.4f}-{lons.max():.4f} E"
        )

        print("Rendering    : raster (imshow, nearest-neighbor)")
        print("Map panel    : square")

        counties_off = args.no_counties or args.no_roads

        print(
            "County lines : "
            + ("OFF" if counties_off else "TIGERweb county boundaries")
        )

        if counties_off:
            counties = []
        else:
            counties = fetch_county_boundaries(bbox)

        print()

        for forecast_minute in ds["forecast_minute"].values.astype(int):
            output_file = (
                output_dir
                / f"hrrr_precip_15min_m{forecast_minute:03d}.png"
            )

            print(
                f"{format_forecast_offset(forecast_minute)} "
                f"(+{forecast_minute:03d} min): "
                f"{output_file.name}"
            )

            plot_one(
                ds,
                forecast_minute,
                output_file,
                counties,
            )

    finally:
        ds.close()

    print()
    print(f"Figures saved under: {output_dir}")


if __name__ == "__main__":
    main()
