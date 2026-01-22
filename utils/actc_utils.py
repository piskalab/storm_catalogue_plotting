import numpy as np
import xarray as xr
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# Colormap definition (exactly as provided)
# -----------------------------------------------------------------------------

ACTC_category_colors = [
    sns.xkcd_rgb['silver'],         # unknown
    sns.xkcd_rgb['reddish brown'],  # surface and subsurface
    sns.xkcd_rgb['white'],          # clear
    sns.xkcd_rgb['dull red'],       # rain in clutter
    sns.xkcd_rgb['off blue'],       # snow in clutter
    sns.xkcd_rgb['dull yellow'],    # cloud in clutter
    sns.xkcd_rgb['dark red'],       # heavy rain
    sns.xkcd_rgb['navy blue'],      # heavy mixed-phase precipitation
    sns.xkcd_rgb['light grey'],     # clear (poss. liquid)
    sns.xkcd_rgb['pale yellow'],    # liquid cloud
    sns.xkcd_rgb['golden'],         # drizzling liquid
    sns.xkcd_rgb['orange'],         # warm rain
    sns.xkcd_rgb['bright red'],     # cold rain
    sns.xkcd_rgb['easter purple'],  # melting snow
    sns.xkcd_rgb['dark sky blue'],  # snow (possible liquid)
    sns.xkcd_rgb['bright blue'],    # snow
    sns.xkcd_rgb['prussian blue'],  # rimed snow (poss. liquid)
    sns.xkcd_rgb['dark teal'],      # rimed snow and SLW
    sns.xkcd_rgb['teal'],           # snow and SLW
    sns.xkcd_rgb['light green'],    # supercooled liquid
    sns.xkcd_rgb['sky blue'],       # ice (poss. liquid)
    sns.xkcd_rgb['bright teal'],    # ice and SLW
    sns.xkcd_rgb['light blue'],     # ice (no liquid)
    sns.xkcd_rgb['pale blue'],      # strat. ice, PSC II
    sns.xkcd_rgb['neon green'],     # PSC Ia
    sns.xkcd_rgb['greenish cyan'],  # PSC Ib
    sns.xkcd_rgb['ugly green'],     # insects
    sns.xkcd_rgb['sand'],           # dust
    sns.xkcd_rgb['pastel pink'],    # sea salt
    sns.xkcd_rgb['dust'],           # continental pollution
    sns.xkcd_rgb['purpley grey'],   # smoke
    sns.xkcd_rgb['dark lavender'],  # dusty smoke
    sns.xkcd_rgb['dusty lavender'], # dusty mix
    sns.xkcd_rgb['pinkish grey'],   # stratospheric aerosol 1 (ash)
    sns.xkcd_rgb['light khaki'],    # stratospheric aerosol 2 (sulphate)
    sns.xkcd_rgb['light grey'],     # stratospheric aerosol 3 (smoke)
]

ACTC_cmap = ListedColormap(ACTC_category_colors)

# -----------------------------------------------------------------------------
# Window loader (same logic as in cpr_utils)
# -----------------------------------------------------------------------------

def _load_window(ds: xr.Dataset, var_name: str, peak_lat: float, half_width: float = 1.5
                 ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[list], Optional[float], Optional[float]]:
    # --- NEW: handle missing dataset ---
    if ds is None:
        return None, None, None, None, None

    # --- NEW: handle missing variable or latitude ---
    if var_name not in ds or 'latitude' not in ds:
        return None, None, None, None, None
    
    lat = ds['latitude'].astype('float64').values
    desired_min = peak_lat - half_width
    desired_max = peak_lat + half_width

    mask = (lat >= desired_min) & (lat <= desired_max)
    if not np.any(mask):
        return None, None, None, None, None

    i0, i1 = np.where(mask)[0][[0, -1]]
    i1 += 1  # include last index

    lat_dim = ds['latitude'].dims[0]
    da = ds[var_name].isel({lat_dim: slice(i0, i1)}).load()

    along_axis = da.get_axis_num(lat_dim)
    arr = da.values
    if along_axis == 0:
        img = arr.T
    else:
        img = arr

    lat_window = lat[i0:i1]

    if lat_window[0] > lat_window[-1]:
        lat_window = lat_window[::-1]
        img = img[:, ::-1]

    dx = np.median(np.diff(lat_window))
    left_edge = lat_window[0] - 0.5 * dx
    right_edge = lat_window[-1] + 0.5 * dx
    extent = [left_edge, right_edge, 0, img.shape[0]]

    center_lat = peak_lat
    return img, center_lat, extent, desired_min, desired_max

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _bounds_for_categories(data: np.ndarray, n_colors: int) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size:
        vmin = int(np.floor(finite.min()))
        vmax = int(np.ceil(finite.max()))
    else:
        vmin, vmax = -1, max(0, n_colors - 2)
    return np.arange(vmin - 0.5, vmax + 1.5, 1)


def plot_actc(row, ax, ds: xr.Dataset, var_name: str = 'synergetic_target_classification') -> bool:
    orbit = row['orbit_frame']
    peak_lat = row['peak_lat']

    out = _load_window(ds, var_name, peak_lat)
    if out[0] is None:
        # Draw empty box
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

        info_text = 'AC-TC Target classification'
        ax.text(0.011, 0.97, info_text, transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        return False

    data, center_lat, extent, desired_min, desired_max = out

    bounds = _bounds_for_categories(data, ACTC_cmap.N)
    norm = BoundaryNorm(bounds, ACTC_cmap.N, clip=False)

    ax.imshow(data, aspect='auto', cmap=ACTC_cmap, interpolation='nearest', norm=norm, extent=extent, origin='upper')

    ax.axvline(x=center_lat, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlim(desired_min, desired_max)

    info_text = 'AC-TC Target classification'
    ax.text(0.011, 0.97, info_text, transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    return True
