import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import hashlib


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

def make_bt_mappable(orbit):
    low, high = _bt_range_for_orbit(orbit)

    cmap = plt.get_cmap("jet_r").copy()
    cmap.set_under(cmap(0.0))   # temperatures below low
    cmap.set_over("white")      # temperatures above high
    cmap.set_bad("white")       # NaN values

    norm = Normalize(
        vmin=low,
        vmax=high,
        clip=False,
    )

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    return mappable

def apply_grayscale_colormap(data):
    cmap = LinearSegmentedColormap.from_list("custom_gray", [(0.2, 0.2, 0.2), (0.95, 0.95, 0.95)])
    norm = plt.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    rgba = cmap(norm(data))
    return rgba[..., :3]

def plot_msi(row, ax, msi):
    orbit = row['earthcare_id']
    peak_lat = row['peak_lat']
    band108 = msi.pixel_values.sel(band="TIR2").values  # 10.8 µm
    band006 = msi.pixel_values.sel(band="VIS").values  # 0.6  µm
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
        # Draw empty box
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

        info_text = 'MSI [no data available]'
        ax.text(0.011, 0.96, info_text, transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        #print(f"[ERROR] No data in ±1.5° latitude range for orbit {orbit}")
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
        info_text = "MSI 10.8µm BT [K]"
        ax.text(0.011, 0.96, info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.text(
            0.01, 0.81, "nadir",
            transform=ax.transAxes,
            va='top', ha='left',
            fontsize=7,
        )

    elif any(letter in orbit for letter in ['D', 'E', 'F']):
        rgb_result  = apply_jet_colormap(band108_masked[rows_sel, :], orbit)
        gray_result = apply_grayscale_colormap(band006_masked[rows_sel, :])
        sandwich    = gray_result * rgb_result

        ax.imshow(np.transpose(sandwich[:, 12:365], (1, 0, 2)), aspect='auto', origin=origin)
        info_text = "MSI 10.8µm BT [K] on 0.6µm background"
        ax.text(0.011, 0.97, info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.text(
            0.01, 0.81, "nadir",
            transform=ax.transAxes,
            va='top', ha='left',
            fontsize=7,
        )

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

    #return True
    bt_mappable = make_bt_mappable(orbit)
    return bt_mappable


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


def plot_summary_horizontal(row, ax, msi, li, time_s=150):
    """
    Swath-style summary plot in MSI index space.
    Geometry matches the MSI panel:
      x = across-track pixel index
      y = along-track index (latitude ordered)

    Shows:
      - MSI grayscale background
      - CPR track as horizontal line
      - LI groups projected into MSI pixel space
    """

    orbit    = row['earthcare_id']
    peak_lat = float(row['peak_lat'])

    try:
        # ------------------------------------------------------------------
        # MSI data
        # ------------------------------------------------------------------
        band108 = msi.pixel_values.sel(band="TIR2").values  # 10.8 µm
        band006 = msi.pixel_values.sel(band="VIS").values  # 0.6  µm
        lat_msi = msi.latitude.values
        lon_msi = msi.longitude.values

        b006_max = np.nanmax(band006)
        band108_masked = np.where(band006 == b006_max, np.nan, band108)
        band006_masked = np.where(band006 == b006_max, np.nan, band006)
        lat_masked     = np.where(band006 == b006_max, np.nan, lat_msi)
        lon_masked     = np.where(band006 == b006_max, np.nan, lon_msi)

        # CPR column index (fixed)
        cpr_col = 278

        # ------------------------------------------------------------------
        # Determine MSI window (±1.5° around peak latitude)
        # ------------------------------------------------------------------
        lat_along = np.where(band006 == b006_max, np.nan, lat_msi)[:, cpr_col]

        lat_mask = (lat_along >= peak_lat - 1.5) & (lat_along <= peak_lat + 1.5)
        if not np.any(lat_mask):
            # Draw empty box
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)

            info_text = 'Lightning clusters over MSI 10.8µm [no MSI data available]'
            ax.text(0.011, 0.97, info_text, transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            #print(f"[SUMMARY] No MSI data in latitude window for orbit {orbit}")
            return False

        rows_sel = np.where(lat_mask)[0]
        lat_sel  = lat_along[rows_sel].astype(float)

        if lat_sel[0] > lat_sel[-1]:
            rows_sel = rows_sel[::-1]
            origin = 'lower'
        else:
            origin = 'upper'

        # plotting
        col_min, col_max = 12, 365
        gray= apply_grayscale_colormap(-band108_masked[rows_sel, :])
        ax.imshow(np.transpose(gray[:, 12:365], (1, 0, 2)), aspect='auto', origin=origin)

        # ------------------------------------------------------------------
        # Add lines
        # ------------------------------------------------------------------
        ax.axhline(y=278 - 12, color='black', linewidth=0.5, linestyle='--')
        center_col = band108_masked[rows_sel, :].shape[0] // 2
        ax.axvline(x=center_col, color='black', linewidth=0.5, linestyle='--')

        # ------------------------------------------------------------------
        # Lightning points within ±time_s
        # ------------------------------------------------------------------
        time_diff_s = (
            li["ec_time_diff"] / pd.Timedelta(seconds=1)
        ).to_numpy()

        time_mask = np.abs(time_diff_s) <= time_s

        if (
            "parallax_corrected_lat" in li.columns
            and "parallax_corrected_lon" in li.columns
        ):
            li_lat = li.loc[time_mask, "parallax_corrected_lat"].to_numpy()
            li_lon = li.loc[time_mask, "parallax_corrected_lon"].to_numpy()
        else:
            li_lat = li.loc[time_mask, "latitude"].to_numpy()
            li_lon = li.loc[time_mask, "longitude"].to_numpy()

        li_cluster = li.loc[time_mask, "cluster_id"].to_numpy()

        # Remove unwanted clusters
        valid_cluster_mask = np.isfinite(li_cluster) & (li_cluster >= 0)

        li_lat = li_lat[valid_cluster_mask]
        li_lon = li_lon[valid_cluster_mask]
        li_cluster = li_cluster[valid_cluster_mask].astype(int)

        # cluster colors (reproducible per orbit)
        seed = int(hashlib.sha256(str(orbit).encode()).hexdigest(), 16) % (2**32)
        label_to_color = _random_cluster_colors(li_cluster, seed)
        colors = np.array([
            label_to_color.get(int(lab), (0.0, 0.0, 0.0))
            if np.isfinite(lab) else (0.0, 0.0, 0.0)
            for lab in li_cluster
        ])

        # ------------------------------------------------------------------
        # Project lightning points into MSI pixel space
        # ------------------------------------------------------------------

        from scipy.spatial import cKDTree

        # Prepare lat lon
        lat_masked_sel = lat_masked[rows_sel, 12:365]
        lon_masked_sel = lon_masked[rows_sel, 12:365]

        # Build KD-tree from *valid* MSI pixels only
        lat_ref = np.deg2rad(peak_lat)
        cosref  = np.cos(lat_ref)

        # Flatten masked MSI grids
        lat_flat = lat_masked_sel.ravel()
        lon_flat = lon_masked_sel.ravel()

        valid = np.isfinite(lat_flat) & np.isfinite(lon_flat)
        if not np.any(valid):
            return True  # nothing to plot

        # Metric-consistent coordinates (degrees)
        msi_y = lat_flat[valid]
        msi_x = lon_flat[valid] * cosref

        msi_points = np.column_stack((msi_x, msi_y))
        tree = cKDTree(msi_points)

        # LI points → same coordinate system
        li_x = li_lon * cosref
        li_y = li_lat
        li_points = np.column_stack((li_x, li_y))

        # Query nearest MSI pixel
        dist_deg, idx = tree.query(li_points, k=1)

        # Keep only LI within ~1 km (≈0.009°)
        deg_1km = 1.0 / 111.0
        keep_dist = dist_deg <= deg_1km

        if np.any(keep_dist):
            idx = idx[keep_dist]
            colors = colors[keep_dist]

            flat_indices = np.flatnonzero(valid)[idx]
            li_rows, li_cols = np.unravel_index(flat_indices, lat_masked_sel.shape)

            # flip y if origin
            x = li_rows
            if origin == 'upper':
                #x = (len(rows_sel)-1) - x
                ax.set_ylim(col_max - col_min, 0)
            else:
                ax.set_ylim(0, col_max - col_min)
            y = li_cols

            ax.scatter(x, y, s=3, c=colors, zorder=3)

        # ------------------------------------------------------------------
        # Formatting
        # ------------------------------------------------------------------
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        #ax.set_ylim(0, col_max - col_min)
        ax.set_xlim(0, len(rows_sel)-1)

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

        info_text = "Lightning clusters over MSI 10.8µm"
        ax.text(
            0.01, 0.97, info_text,
            transform=ax.transAxes,
            va='top', ha='left',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        ax.text(
            0.01, 0.81, "nadir",
            transform=ax.transAxes,
            va='top', ha='left',
            fontsize=7,
        )

        return True

    except Exception as e:
        print(f"[SUMMARY ERROR] Orbit {orbit}: {e}")
        return False
