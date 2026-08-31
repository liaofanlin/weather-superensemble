#!/usr/bin/env python3
"""
plot_hrrr_precip.py

Clean HRRR precipitation plot with ONLY selected major Denver highways.

Road source:
    U.S. Census TIGERweb Transportation service

Only these routes are retained:
    I-25
    I-70
    I-76
    I-225
    I-270
    US-6
    US-36
    US-285

All retained roads are black and each route is labeled once.
No local roads, no state highways, no business routes, no express-lane labels.

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


TIGER_BASE = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
    "TIGERweb/Transportation/MapServer"
)

# Query both layers, then FILTER strictly by route name.
ROAD_LAYERS = [
    (2, "primary"),
    (6, "secondary"),
]

# Exact routes we want on the map.
ALLOWED_ROUTES = {
    "I-25",
    "I-70",
    "I-76",
    "I-225",
    "I-270",
    "US-6",
    "US-36",
    "US-285",
}


def normalize_route_name(name):
    """
    Convert TIGER road names such as:
        'I-25'
        'I 25'
        'Interstate 25'
        'US Hwy 36'
        'US Highway 36'
        'U.S. Hwy 36'
    to canonical labels such as I-25 or US-36.

    Business routes, express lanes, and other variants are rejected.
    """
    if not name:
        return None

    s = str(name).upper().strip()
    s = re.sub(r"\s+", " ", s)

    # Reject variants we do not want.
    reject_words = [
        "BUS",
        "BUSINESS",
        "EXPRESS",
        "TOLL",
        "FRONTAGE",
        "RAMP",
        "SPUR",
        "LOOP",
        "CONNECTOR",
    ]
    if any(word in s for word in reject_words):
        return None

    # Interstate
    m = re.search(r"(?:INTERSTATE|I)[\s\-]*([0-9]{1,3})\b", s)
    if m:
        route = f"I-{int(m.group(1))}"
        return route if route in ALLOWED_ROUTES else None

    # U.S. Highway
    s2 = (
        s.replace("U.S.", "US")
         .replace("U.S", "US")
         .replace("UNITED STATES", "US")
    )

    m = re.search(
        r"\bUS(?: HWY| HIGHWAY)?[\s\-]*([0-9]{1,3})\b",
        s2,
    )
    if m:
        route = f"US-{int(m.group(1))}"
        return route if route in ALLOWED_ROUTES else None

    return None


def query_tiger_layer(layer_id, bbox):
    south, west, north, east = bbox
    url = f"{TIGER_BASE}/{layer_id}/query"

    params = {
        "where": "1=1",
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "BASENAME,NAME,RTTYP,MTFCC",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }

    r = requests.get(
        url,
        params=params,
        timeout=60,
        headers={"User-Agent": "HRRR-Denver-major-highways/1.0"},
    )
    r.raise_for_status()
    return r.json()


def fetch_major_highways(bbox):
    """
    Download TIGERweb geometry and retain ONLY the allow-listed routes.
    """
    roads = []

    print("Downloading selected major highways from TIGERweb...")

    for layer_id, layer_name in ROAD_LAYERS:
        try:
            payload = query_tiger_layer(layer_id, bbox)
        except Exception as exc:
            print(f"  WARNING: {layer_name} layer failed: {exc}")
            continue

        kept = 0

        for feature in payload.get("features", []):
            props = feature.get("properties") or {}
            geom = feature.get("geometry") or {}

            # Try NAME first, then BASENAME.
            route = normalize_route_name(props.get("NAME"))
            if route is None:
                route = normalize_route_name(props.get("BASENAME"))

            if route is None:
                continue

            coords = geom.get("coordinates")
            geom_type = geom.get("type")

            if not coords:
                continue

            if geom_type == "LineString":
                parts = [coords]
            elif geom_type == "MultiLineString":
                parts = coords
            else:
                continue

            for part in parts:
                if len(part) < 2:
                    continue

                roads.append(
                    {
                        "route": route,
                        "lons": [p[0] for p in part],
                        "lats": [p[1] for p in part],
                    }
                )
                kept += 1

        print(f"  {layer_name:9s}: retained {kept} pieces")

    # Remove exact duplicate geometries that can occur between layers.
    unique = []
    seen = set()

    for road in roads:
        key = (
            road["route"],
            round(road["lons"][0], 6),
            round(road["lats"][0], 6),
            round(road["lons"][-1], 6),
            round(road["lats"][-1], 6),
            len(road["lons"]),
        )

        reverse_key = (
            road["route"],
            round(road["lons"][-1], 6),
            round(road["lats"][-1], 6),
            round(road["lons"][0], 6),
            round(road["lats"][0], 6),
            len(road["lons"]),
        )

        if key in seen or reverse_key in seen:
            continue

        seen.add(key)
        unique.append(road)

    roads = unique

    routes = sorted({r["route"] for r in roads})

    print(f"Highway pieces loaded : {len(roads)}")
    print(
        "Routes loaded         : "
        + (", ".join(routes) if routes else "NONE")
    )

    return roads


def draw_highways(ax, roads, transform=None):
    if not roads:
        return

    # Draw only black highway centerlines.
    for road in roads:
        kwargs = {
            "color": "black",
            "linewidth": 1.8,
            "alpha": 1.0,
            "zorder": 20,
            "solid_capstyle": "round",
            "solid_joinstyle": "round",
        }

        if transform is not None:
            kwargs["transform"] = transform

        ax.plot(
            road["lons"],
            road["lats"],
            **kwargs,
        )

    # One label per route: use longest retained piece.
    representative = {}

    for road in roads:
        lon = np.asarray(road["lons"])
        lat = np.asarray(road["lats"])

        length = np.sum(
            np.sqrt(
                np.diff(lon) ** 2
                + np.diff(lat) ** 2
            )
        )

        route = road["route"]

        if length > representative.get(route, (0.0, None))[0]:
            representative[route] = (length, road)

    for route in sorted(representative):
        _, road = representative[route]

        mid = len(road["lons"]) // 2

        kwargs = {
            "fontsize": 8,
            "fontweight": "bold",
            "color": "black",
            "ha": "center",
            "va": "center",
            "zorder": 22,
            "bbox": {
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        }

        if transform is not None:
            kwargs["transform"] = transform

        ax.text(
            road["lons"][mid],
            road["lats"][mid],
            route,
            **kwargs,
        )


def format_time(value):
    try:
        if np.isnat(value):
            return None
    except TypeError:
        pass

    return np.datetime_as_string(value, unit="m")


def plot_one(ds, fhr, output_file, roads):
    field = ds["precip_1h_mm"].sel(forecast_hour=fhr)

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    values = field.values

    lon2d, lat2d = np.meshgrid(lons, lats)

    # Precipitation thresholds converted exactly from inches to millimeters.
    # Original inch thresholds:
    # 0.002, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 2.00 in
    levels = np.array([
        0.0508,   # 0.002 in
        0.2540,   # 0.01 in
        1.2700,   # 0.05 in
        2.5400,   # 0.10 in
        6.3500,   # 0.25 in
        12.7000,  # 0.50 in
        19.0500,  # 0.75 in
        25.4000,  # 1.00 in
        50.8000,  # 2.00 in
    ])

    # Match the supplied precipitation color bar:
    # gray -> cyan -> green -> lime -> yellow -> amber -> orange -> red-orange
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
        f"HRRR 1-h precipitation: F{fhr:02d}\n"
        f"Accumulation F{fhr-1:02d}-F{fhr:02d}"
    )

    init_time = ds.attrs.get("initialization_time")

    if init_time:
        init_time = str(init_time).replace(".000000000", "")
        title += f" | Init {init_time}"

    if "valid_time" in ds:
        valid_time = format_time(
            ds["valid_time"].sel(forecast_hour=fhr).values
        )

        if valid_time:
            title += f"\nValid {valid_time} UTC"

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(9, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())

        pcm = ax.contourf(
            lon2d,
            lat2d,
            values,
            levels=levels,
            cmap=precip_cmap,
            norm=precip_norm,
            extend="both",
            transform=ccrs.PlateCarree(),
            zorder=1,
        )

        ax.add_feature(
            cfeature.STATES,
            linewidth=0.7,
            zorder=4,
        )

        ax.set_extent(
            [
                float(lons.min()),
                float(lons.max()),
                float(lats.min()),
                float(lats.max()),
            ],
            crs=ccrs.PlateCarree(),
        )

        draw_highways(
            ax,
            roads,
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

        cbar = fig.colorbar(
            pcm,
            ax=ax,
            pad=0.03,
        )

    except ImportError:
        fig, ax = plt.subplots(figsize=(9, 7))

        pcm = ax.contourf(
            lon2d,
            lat2d,
            values,
            levels=levels,
            cmap=precip_cmap,
            norm=precip_norm,
            extend="both",
            zorder=1,
        )

        draw_highways(ax, roads)

        ax.set_xlim(
            float(lons.min()),
            float(lons.max()),
        )
        ax.set_ylim(
            float(lats.min()),
            float(lats.max()),
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        ax.grid(
            True,
            linewidth=0.4,
            alpha=0.4,
            zorder=2,
        )

        cbar = fig.colorbar(
            pcm,
            ax=ax,
            pad=0.02,
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
    cbar.set_label("1-h precipitation (mm)")
    ax.set_title(title)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

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
        "--no-roads",
        action="store_true",
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
        print("Plot regridded HRRR precipitation")
        print("=" * 72)

        print(f"Input file   : {input_file}")
        print(f"Grid         : {len(lats)} x {len(lons)}")

        print(
            f"Domain       : "
            f"{lats.min():.4f}-{lats.max():.4f} N, "
            f"{lons.min():.4f}-{lons.max():.4f} E"
        )

        print(
            "Highways     : "
            + (
                "OFF"
                if args.no_roads
                else "I-25, I-70, I-76, I-225, I-270, "
                     "US-6, US-36, US-285"
            )
        )

        if args.no_roads:
            roads = []
        else:
            roads = fetch_major_highways(bbox)

        print()

        for fhr in ds["forecast_hour"].values.astype(int):

            output_file = (
                output_dir
                / f"hrrr_precip_1h_f{fhr:02d}.png"
            )

            print(
                f"F{fhr:02d}: "
                f"{output_file.name}"
            )

            plot_one(
                ds,
                fhr,
                output_file,
                roads,
            )

    finally:
        ds.close()

    print()
    print(f"Figures saved under: {output_dir}")


if __name__ == "__main__":
    main()
