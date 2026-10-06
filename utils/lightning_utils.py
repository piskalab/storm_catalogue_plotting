import os
from executing import Source
import numpy as np
import pandas as pd
import xarray as xr
import json
from sklearn.metrics.pairwise import haversine_distances


def get_lightning_counts(row, file_list_li, track_ds, count_mode: str = "strict"):
    """
    Returns:
      - plot_lats (ascending): CPR latitudes within ±1.5° of peak_lat
      - plot_counts: overall LI group counts per CPR point (strict/loose)
      - plot_counts_cluster: per-subcluster LI group counts per CPR point
      - desired_min, desired_max: numeric bounds for plotting
    """
    orbit         = row['orbit_frame']
    #orbit         = row['earthcare_id']
    cluster_id = int(row['cluster_id'])
    peak_lat      = float(row['peak_lat'])

    try:
        cpr_lat = np.asarray(track_ds['latitude'].values, dtype=float)

        # --- count mode ---
        if count_mode not in {"strict", "loose"}:
            print(f"[WARN] Unknown count_mode '{count_mode}', defaulting to 'strict'")
            count_mode = "strict"

        varname = 'lightning_count_2p5' if count_mode == "strict" else 'lightning_count_5'

        if varname not in track_ds.variables:
            print(f"[WARN] Requested {count_mode} counts not found in track data for orbit {orbit}")
            return None, None, None, None, None

        # --- per-subcluster counts ---
        if cluster_id not in track_ds['cluster_id'].values:
            print(f"[WARN] Cluster {cluster_id} not found in track counts for orbit {orbit}")
            return None, None, None, None, None

        per_cluster_counts = (
            track_ds[varname]
            .sel(cluster_id=cluster_id)
            .values
            .astype(np.int64)
        )

        # --- overall counts = sum over subclusters ---
        overall_counts = (
            track_ds[varname]
            .sum(dim="cluster_id")
            .values
            .astype(np.int64)
        )

        # --- latitude window ---
        half_width  = 1.5
        desired_min = peak_lat - half_width
        desired_max = peak_lat + half_width

        mask_lat = (cpr_lat >= desired_min) & (cpr_lat <= desired_max)
        if not np.any(mask_lat):
            print(f"[WARN] No CPR data in ±1.5° around peak_lat for orbit {orbit}")
            return None, None, None, None, None

        plot_lats           = cpr_lat[mask_lat]
        plot_counts         = overall_counts[mask_lat]
        plot_counts_cluster = per_cluster_counts[mask_lat]

        # --- ensure ascending latitude ---
        if plot_lats[0] > plot_lats[-1]:
            plot_lats           = plot_lats[::-1]
            plot_counts         = plot_counts[::-1]
            plot_counts_cluster = plot_counts_cluster[::-1]

        return plot_lats, plot_counts, plot_counts_cluster, desired_min, desired_max

    except Exception as e:
        print(f"[ERROR] Failed to read precomputed counts for orbit {orbit}: {e}")
        return None, None, None, None, None
    

def plot_lightning_info(row, ax, plot_lats, plot_counts, plot_counts_cluster,
                        desired_min=None, desired_max=None,
                        time_s=150, distance_km=2.5):
    """
    Plot lightning counts along-track. Shows empty space to the requested bounds by xlim().
    Assumes plot_lats is ascending (if using get_lightning_counts above).
    """
    if plot_lats is None or len(plot_lats) == 0:
        return False

    orbit        = row['orbit_frame']
    #orbit        = row['earthcare_id']
    surface_type = row['surface_type']
    datetime_str = row['peak_datetime'].strftime('%Y-%m-%d %H:%M')
    peak_lat     = float(row['peak_lat'])
    peak_lon     = float(row['peak_lon'])
    mean_dist_km = float(row['cluster_dist_km'])
    cluster_li_c = int(row['cluster_lightning'])
    close_li_c   = int(row['nadir_lightning'])
    peak_li_c    = int(row['peak_lightning'])
    source       = row['source']

    # If not provided, default to the tight data range; otherwise use the desired ±1.5° window
    if desired_min is None or desired_max is None:
        desired_min = float(plot_lats[0])
        desired_max = float(plot_lats[-1])

    # y-limits with a small headroom
    ymax = max(
        1,
        np.nanmax([np.nanmax(plot_counts), np.nanmax(plot_counts_cluster)]) + 10
    )
    ax.set_ylim(1, ymax)

    # Always show the full requested window — this creates empty space if data stop early
    ax.set_xlim(desired_min, desired_max)

    # Lines
    ax.plot(plot_lats, plot_counts,          lw=1.5, color='black', zorder=2)
    ax.plot(plot_lats, plot_counts_cluster,  lw=1.5, color='red',   zorder=3)

    # Vertical guide at the true peak latitude (not midpoint)
    ax.axvline(x=peak_lat, color='black', linestyle='--', linewidth=0.5)

    # Clean aesthetics
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Centered info column using a twin axis (same y-scale)
    ax_center = ax.twinx()
    ax_center.spines['left'].set_position(('axes', 0.5))
    ax_center.spines['left'].set_visible(False)
    ax_center.spines['right'].set_visible(False)
    ax_center.spines['top'].set_visible(False)
    ax_center.spines['bottom'].set_visible(False)
    ax_center.yaxis.set_ticks_position('left')
    ax_center.tick_params(axis='y', direction='in', colors='gray', labelsize=7, pad=-18)
    ax_center.set_ylim(ax.get_ylim())

    info_text = (
        f"Number of lightning groups around nadir track \n(±{(time_s/60)} min, ±{distance_km:.1f} km)\n\n"
        f"⚡ Storm summary\n"
        f"EarthCARE ID: {orbit}\n"
        f"Lightning source: {source} / Surface type: {surface_type}\n"
        f"Time: {datetime_str}\n"
        f"Dist. of nadir from storm center: {mean_dist_km:.1f} km\n"
        #f"Storm counts (±{(time_s/60)} min): {cluster_li_c} (all), {close_li_c} (±{distance_km:.1f} km)\n"
        #f"Peak counts (±{(time_s/60)} min, ±{distance_km:.1f} km): {peak_li_c}\n"
        f"Nadir lightning counts: {close_li_c}\n"
        f"Peak lightning counts: {peak_li_c}\n"
        f"Peak lon/lat: {peak_lon:.2f}°, {peak_lat:.2f}°"
    )

    ax.text(0.011, 0.96, info_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    return True
