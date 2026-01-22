import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import hashlib

from config import LI_PATH


def _bt_range_for_orbit(orbit: str):
    if any(letter in orbit for letter in ['A', 'E']):
        return 180.0, 230.0
    if any(letter in orbit for letter in ['B', 'D', 'F', 'H']):
        return 200.0, 240.0
    # fallback if no letter matched
    return 200.0, 240.0

def apply_jet_colormap(bt_K: np.ndarray, orbit: str) -> np.ndarray:
    """
    Map brightness temperature (K) to an RGB image using 'jet' between a
    per-orbit [low, high] range. Below 'low' => use jet(0.0) edge color.
    Above 'high' => neutral white so grayscale background is preserved when multiplied.

    Returns float RGB array in [0,1], shape (H,W,3).
    """
    low, high = _bt_range_for_orbit(orbit)
    cmap = plt.get_cmap('jet_r')
    # Start neutral (white) so gray*rgb leaves gray unchanged where we don't colorize
    rgb = np.ones(bt_K.shape + (3,), dtype=float)
    # Handle NaNs: keep them white so they don't darken the grayscale
    valid = np.isfinite(bt_K)
    # Below low -> edge color (jet(0.0))
    low_edge_rgb = np.array(cmap(0.0)[:3])
    mask_low = valid & (bt_K <= low)
    rgb[mask_low] = low_edge_rgb
    # Between low and high -> normalized jet
    mask_in = valid & (bt_K > low) & (bt_K <= high)
    if np.any(mask_in):
        norm = (bt_K[mask_in] - low) / (high - low)
        rgb[mask_in] = cmap(norm)[:, :3]
    return rgb

def apply_grayscale_colormap(data):
    cmap = LinearSegmentedColormap.from_list("custom_gray", [(0.2, 0.2, 0.2), (0.95, 0.95, 0.95)])
    norm = plt.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    rgba = cmap(norm(data))
    return rgba[..., :3]

def plot_msi(row, ax, msi):
    orbit = row['orbit_frame']
    peak_lat = row['peak_lat']
    band108 = msi.pixel_values.values[5, :, :]
    band006 = msi.pixel_values.values[0, :, :]
    latitude = msi.latitude.values

    band108_masked = np.where(band006 == np.max(band006), np.nan, band108)
    band006_masked = np.where(band006 == np.max(band006), np.nan, band006)
    lat_masked     = np.where(band006 == np.max(band006), np.nan, latitude)

    # cpr is in index 278 (with nan values)
    cpr_col = 278
    lat_along = lat_masked[:, cpr_col]

    # select rows in ±1.5° around peak
    lat_range_mask = (lat_along >= (peak_lat - 1.5)) & (lat_along <= (peak_lat + 1.5))
    if not np.any(lat_range_mask):
        print(f"[ERROR] No data in ±1.5° latitude range for orbit {orbit}")
        return False

    # --- minimal change block: ensure lat is ascending within the window ---
    rows_sel = np.where(lat_range_mask)[0]
    lat_sel  = lat_along[rows_sel].astype(float)
    if lat_sel[0] > lat_sel[-1]:
        rows_sel = rows_sel[::-1]
        origin = 'lower'
    else:
        origin = 'upper'
    # ----------------------------------------------------------------------

    if any(letter in orbit for letter in ['A', 'B', 'H']):
        #rgb_result = band108_masked[rows_sel, :]
        rgb_result  = apply_jet_colormap(band108_masked[rows_sel, :], orbit)
        gray_result = apply_grayscale_colormap(-band108_masked[rows_sel, :])
        sandwich    = gray_result * rgb_result
        #ax.imshow(rgb_result[:, 12:365].T, aspect='auto', cmap=jet_greys)
        ax.imshow(np.transpose(sandwich[:, 12:365], (1, 0, 2)), aspect='auto', origin=origin)
        info_text = "MSI enhanced 10.8µm"
        ax.text(0.011, 0.97, info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    elif any(letter in orbit for letter in ['D', 'E', 'F']):
        rgb_result  = apply_jet_colormap(band108_masked[rows_sel, :], orbit)
        gray_result = apply_grayscale_colormap(band006_masked[rows_sel, :])
        sandwich    = gray_result * rgb_result

        ax.imshow(np.transpose(sandwich[:, 12:365], (1, 0, 2)), aspect='auto', origin=origin)
        info_text = "MSI enhanced 10.8µm on 0.6µm background"
        ax.text(0.011, 0.97, info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.axhline(y=278 - 12, color='black', linewidth=0.5, linestyle='--')
    center_col = band108_masked[rows_sel, :].shape[0] // 2
    ax.axvline(x=center_col, color='black', linewidth=0.5, linestyle='--')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    return True


def _random_cluster_colors(labels, seed):
    """
    Create an RGB color for each unique cluster label (deterministic given seed).
    Returns:
        label_to_color: dict {label: (r,g,b)}
    """
    uniq = np.unique(labels[~np.isnan(labels)])
    uniq = uniq.astype(int)
    rng = np.random.default_rng(seed)
    # Random RGB colors in [0,1]
    colors = rng.random((len(uniq), 3))
    # (Optional) avoid very light colors; uncomment if needed:
    # colors = 0.85 * colors
    return {lab: tuple(colors[i]) for i, lab in enumerate(uniq)}


def plot_summary(row, ax, msi, li, cpr, time_s=150):
    """
    Plots a map showing MSI (grayscale), CPR track, and LI groups by cluster.
    Grayscale rules:
      - Frames A/B/H: use band108_masked with REVERSED grayscale (cold→bright)
      - Frames D/E/F: use band006_masked with normal grayscale
      - vmin/vmax computed from in-window data (robust percentiles)
    """
    orbit    = row['orbit_frame']
    peak_lat = float(row['peak_lat'])
    peak_lon = float(row['peak_lon'])

    try:
        # --- MSI data ---
        band108 = msi.pixel_values.values[5, :, :]  # 10.8 µm
        band006 = msi.pixel_values.values[0, :, :]  # 0.6  µm
        lat_msi = msi.latitude.values
        lon_msi = msi.longitude.values

        b006_max = np.nanmax(band006)
        band108_masked = np.where(band006 == b006_max, np.nan, band108)
        band006_masked = np.where(band006 == b006_max, np.nan, band006)

        # --- CPR window (±1.5°) ---
        cpr_lat = np.asarray(cpr['latitude'].values, dtype=float)
        cpr_lon = np.asarray(cpr['longitude'].values, dtype=float)
        desired_min = peak_lat - 1.5
        desired_max = peak_lat + 1.5
        cpr_mask = (cpr_lat >= desired_min) & (cpr_lat <= desired_max)
        cpr_lat_subset = cpr_lat[cpr_mask]
        cpr_lon_subset = cpr_lon[cpr_mask]
        if cpr_lat_subset.size == 0:
            print(f"[SUMMARY] No CPR data in latitude window for orbit {orbit}")
            return False

        # (kept) pad edges for CPR path continuity
        pad_left = (cpr_lat_subset[0]  > desired_min)
        pad_right = (cpr_lat_subset[-1] < desired_max)
        if pad_left or pad_right:
            cpr_lat_subset = np.concatenate(([desired_min] if pad_left else [], cpr_lat_subset,
                                                [desired_max] if pad_right else []))
            cpr_lon_subset = np.concatenate(([np.nan] if pad_left else [], cpr_lon_subset,
                                                [np.nan] if pad_right else []))

        # --- LI points within ±time_s seconds ---
        time_mask = np.abs(li['ec_time_diff'].values / np.timedelta64(1, 's')) <= time_s
        if "parallax_corrected_lat" in li and "parallax_corrected_lon" in li:
            li_lat = li["parallax_corrected_lat"].values[time_mask]
            li_lon = li["parallax_corrected_lon"].values[time_mask]
        else:
            li_lat = li["latitude"].values[time_mask]
            li_lon = li["longitude"].values[time_mask]
        li_cluster = li['subcluster_id'].values[time_mask]
        

        seed = int(hashlib.sha256(str(orbit).encode()).hexdigest(), 16) % (2**32)
        label_to_color = _random_cluster_colors(li_cluster, seed)
        colors = np.array([label_to_color.get(int(lab), (0.0, 0.0, 0.0))
                            if np.isfinite(lab) else (0.0, 0.0, 0.0)
                            for lab in li_cluster], dtype=float)

        # --- Map extent ---
        lat_min = float(np.nanmin(cpr_lat_subset))
        lat_max = float(np.nanmax(cpr_lat_subset))

        # restrict MSI longitudes to the lat window to determine lon extent
        lat_window_mask = (lat_msi >= desired_min) & (lat_msi <= desired_max)
        selected_lon_data = lon_msi[lat_window_mask]
        if selected_lon_data.size == 0:
            print(f"[SUMMARY] No MSI data in latitude window for orbit {orbit}")
            return False
        lon_min = float(np.nanmin(selected_lon_data))
        lon_max = float(np.nanmax(selected_lon_data))

        # --- Choose band by frame ---
        use_band108 = any(letter in orbit for letter in ['A', 'B', 'H'])
        band_src = band108_masked if use_band108 else band006_masked

        # --- Scale using only in-window data (robust percentiles) ---
        in_window = ((lat_msi >= desired_min) & (lat_msi <= desired_max) &
                        (lon_msi >= lon_min)     & (lon_msi <= lon_max))
        vals = band_src[in_window]
        vmin, vmax = np.nanmin(band_src), np.nanmax(band_src)

        # Mask out-of-window pixels so they don't render off-extent
        band_plot = np.where(in_window, band_src, np.nan)

        # --- Grayscale colormap (and reversed for thermal) ---
        cmap_gray = LinearSegmentedColormap.from_list(
            "custom_gray", [(0.2, 0.2, 0.2), (0.95, 0.95, 0.95)]
        )
        cmap_use = cmap_gray.reversed() if use_band108 else cmap_gray

        # --- Plot ---
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        ax.pcolormesh(
            lon_msi, lat_msi, band_plot,
            cmap=cmap_use, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), shading='auto', zorder=1
        )

        # CPR swath
        ax.plot(cpr_lon_subset, cpr_lat_subset, color='blue', linewidth=1,
                transform=ccrs.PlateCarree(), zorder=2)

        # LI clusters
        ax.scatter(li_lon, li_lat, s=2, c=colors,
                    transform=ccrs.PlateCarree(), zorder=3)

        # Grid / ticks
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        tick_label_style = {'size': 7, 'color': 'gray'}
        gl.xlabel_style = tick_label_style
        gl.ylabel_style = tick_label_style
        gl.xlocator = mticker.MultipleLocator(1)
        gl.ylocator = mticker.MultipleLocator(1)
        ax.tick_params(axis='both', direction='out', length=3, colors='gray', labelsize=7)

        return True

    except Exception as e:
        print(f"[SUMMARY ERROR] Orbit {orbit}: {e}")
        return False