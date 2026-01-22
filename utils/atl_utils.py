import numpy as np
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl

def register_rgb_colormap(r, g, b, colormap_name):

    # Ensure R, G, B arrays are of the same length
    assert len(r) == len(g) == len(b), "R, G, B arrays must have the same length."

    # Stack R, G, B arrays horizontally and normalize to the range [0, 1]
    rgb_values = np.vstack((r, g, b)).T / 255.0

    # Create the colormap
    colormap = ListedColormap(rgb_values, name=colormap_name)

    # Background color
    colormap.set_bad("white")

    return register_colormap(colormap)

def register_colormap(colormap, overwrite=True):
   
    # Check if the colormap is already registered
    if colormap.name in plt.colormaps():
        if not overwrite:
            print(f"Colormap '{colormap.name}' is already registered. Skipping registration.")
            return plt.get_cmap(colormap.name)
        else:
            print(f"Colormap '{colormap.name}' is already registered. Overwriting by default.")

    # Register the colormap
    mpl.colormaps.register(cmap=colormap, force=True)

    return colormap


def define_calipso_cm():
    #Predefined list of RGB values
    red=(0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
         0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,\
         255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,\
         255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,\
         255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255, 70, 70,\
         70, 90, 90, 90,110,110,110,130,130,130,150,150,150,150,150,150,170,170,170,170,\
         170,170,180,180,180,180,180,180,190,190,190,190,190,190,200,200,200,200,200,200,\
         210,210,210,210,210,210,215,215,215,215,215,215,220,220,220,220,220,220,225,225,\
         225,225,225,225,230,230,230,230,230,230,235,235,235,235,235,235,240,240,240,240,\
         240,240,240,245,245,245,245,245,245,245,255,255,255,255,255,255,255)


    green=(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,\
           42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,127,127,127,127,127,127,\
           127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,\
           127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,255,255,255,127,\
           127,127,170,170,170,170,255,255,255,255,255,255,255,255,255,255,255,255,255,255,\
           255,230,230,230,230,230,230,230,230,230,212,212,212,212,212,212,212,212,212,170,\
           170,170,170,170,170,170,170,170,127,127,127,127,127,127, 85, 85, 85, 85, 85, 85,\
           0,  0,  0,  0,  0,  0, 42, 42, 42, 42, 42, 42, 85, 85, 85,127,127,127, 70, 70,\
           70, 90, 90, 90,110,110,110,130,130,130,150,150,150,150,150,150,170,170,170,170,\
           170,170,180,180,180,180,180,180,190,190,190,190,190,190,200,200,200,200,200,200,\
           210,210,210,210,210,210,215,215,215,215,215,215,220,220,220,220,220,220,225,225,\
           225,225,225,225,230,230,230,230,230,230,235,235,235,235,235,235,240,240,240,240,\
           240,240,240,245,245,245,245,245,245,245,255,255,255,255,255,255,255)

    blue=(170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,\
          170,170,170,170,170,170,170,170,170,170,170,170,170,170,255,255,255,255,255,255,\
          255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,\
          255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,170,170,170,127,\
          127,127, 85, 85, 85, 85,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
          0,  0,  0,  0,  0,  0, 85, 85, 85, 85, 85, 85,127,127,127,170,170,170, 70, 70,\
          70, 90, 90, 90,110,110,110,130,130,130,150,150,150,150,150,150,170,170,170,170,\
          170,170,180,180,180,180,180,180,190,190,190,190,190,190,200,200,200,200,200,200,\
          210,210,210,210,210,210,215,215,215,215,215,215,220,220,220,220,220,220,225,225,\
          225,225,225,225,230,230,230,230,230,230,235,235,235,235,235,235,240,240,240,240,\
          240,240,240,245,245,245,245,245,245,245,255,255,255,255,255,255,255)
    
    # Register the colormap
    return register_rgb_colormap(red, green, blue, 'calipso')

calipso_cm = define_calipso_cm()


def define_aebd_cmaps():
    """
    Two-layer colormaps:
      - blue/gray 'bone' style for classes 1,2,4
      - orange/brown for classes 3,5
    NaNs are fully transparent.
    """
    bone = plt.cm.bone
    cmap_blue = mcolors.LinearSegmentedColormap.from_list(
        "bone_trimmed",
        bone(np.linspace(0.3, 1.0, 256))  # skip darkest 30%
    )

    cmap_orange = mcolors.LinearSegmentedColormap.from_list(
        "orange_brown",
        [
            "#ffe5b4",  # light peach/orange
            "#ff8c00",  # strong orange
            "#8b4513",  # dark brown
        ]
    )

    # Make NaNs fully transparent
    cmap_blue = cmap_blue.copy()
    cmap_orange = cmap_orange.copy()
    cmap_blue.set_bad(alpha=0)
    cmap_orange.set_bad(alpha=0)

    return cmap_blue, cmap_orange


aebd_cmap_blue, aebd_cmap_orange = define_aebd_cmaps()

def _load_window(ds, var_name, peak_lat, half_width=1.5):
    lat = ds['ellipsoid_latitude'].astype('float64').values
    #lat = ds['ellipsoid_latitude'].astype('float64').values
    desired_min = peak_lat - half_width
    desired_max = peak_lat + half_width

    # mask & indices
    mask = (lat >= desired_min) & (lat <= desired_max)
    if not np.any(mask):
        return None, None, None, None, None

    i0, i1 = np.where(mask)[0][[0, -1]]
    i1 += 1  # include last index

    lat_dim = ds['ellipsoid_latitude'].dims[0]
    #lat_dim = ds['ellipsoid_latitude'].dims[0]
    da = ds[var_name].isel({lat_dim: slice(i0, i1)}).load()

    # Ensure latitude is on x-axis (columns)
    along_axis = da.get_axis_num(lat_dim)
    arr = da.values
    if along_axis == 0:
        img = arr.T
    else:
        img = arr

    lat_window = lat[i0:i1]

    # Optional: make latitude increasing left→right
    if lat_window[0] > lat_window[-1]:
        lat_window = lat_window[::-1]
        img = img[:, ::-1]

    # compute edges from centers
    dx = np.median(np.diff(lat_window))
    left_edge  = lat_window[0]  - 0.5*dx
    right_edge = lat_window[-1] + 0.5*dx
    extent = [left_edge, right_edge, 0, img.shape[0]]

    # find center column (in lat units now)
    center_lat = peak_lat

    return img, center_lat, extent, desired_min, desired_max

def plot_atlid(row, ax, ds):
    orbit = row.get('orbit_frame', None)
    peak_lat = row['peak_lat']

    # ---- load main data window (same helper as CPR) ----
    out_bs = _load_window(ds, 'mie_attenuated_backscatter', peak_lat)
    if out_bs[0] is None:
        return False

    data, center_lat, extent, desired_min, desired_max = out_bs

    # ---- masking & split by class ----
    vmin, vmax = 1e-7, 1e-5
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Replace NaN/inf with 1e-7
    data = np.where(~np.isfinite(data), 1e-7, data)
    # Values below threshold also mapped to 1e-7
    data = np.where(data <= vmin, 1e-7, data)

    if np.all(~np.isfinite(data)):
        return False

    # ---- plotting ----
    ax.imshow(
        data,
        aspect='auto',
        cmap=calipso_cm,
        #cmap='Blues_r',
        norm=norm,
        extent=extent,
        origin='upper',
        interpolation='nearest',
    )

    # vertical line at peak lat
    ax.axvline(x=center_lat, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlim(desired_min, desired_max)

    units = getattr(ds['mie_attenuated_backscatter'], "units", "unknown")

    info_text = f"ATLID Mie attenuated backscatter [{units}]"
    ax.text(
        0.011, 0.97,
        info_text,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=8,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # same minimalist axes as CPR plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    return True


def plot_atlid_ebd(row, ax, AEBD):
    """
    CPR-style panel for ATLID particle_backscatter_coefficient_355nm,
    using AEBD.simple_classification to split colors:

      - classes 1,2,4 → bone-like cmap (blue/gray)
      - classes 3,5   → orange/brown cmap

    Log-scaled range [1e-6, 1e-3].
    """

    # keep same interface pattern as plot_reflectivity
    orbit = row.get('orbit_frame', None)
    peak_lat = row['peak_lat']

    # ---- load main data window (same helper as CPR) ----
    out_bs = _load_window(AEBD, 'particle_backscatter_coefficient_355nm', peak_lat)
    if out_bs[0] is None:
        return False

    data, center_lat, extent, desired_min, desired_max = out_bs

    # ---- load classification window, using the same helper ----
    out_cls = _load_window(AEBD, 'simple_classification', peak_lat)
    if out_cls[0] is None:
        return False

    cls_data = out_cls[0]

    # ensure shapes match (they should, but guard anyway)
    if cls_data.shape != data.shape:
        # try basic broadcasting along vertical dimension if needed
        try:
            cls_data = np.broadcast_to(cls_data, data.shape)
        except ValueError:
            # cannot align → skip plotting
            return False

    # ---- masking & split by class ----
    vmin, vmax = 1e-6, 1e-3
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # base mask: invalid or below threshold
    base_mask = ~np.isfinite(data)
    base_mask |= (data <= vmin)

    cls_int = cls_data.astype(int)

    # classes 1,2,4 → blue-ish
    mask_blue = base_mask | ~np.isin(cls_int, [1, 2, 4])
    # classes 3,5 → orange/brown
    mask_orange = base_mask | ~np.isin(cls_int, [3, 5])

    data_blue   = np.where(mask_blue,   np.nan, data)
    data_orange = np.where(mask_orange, np.nan, data)

    if np.all(np.isnan(data_blue)) and np.all(np.isnan(data_orange)):
        # nothing to plot
        return False

    # ---- plotting (same style as plot_reflectivity) ----

    # layer 1: bone-like for classes 1,2,4
    ax.imshow(
        data_blue,
        aspect='auto',
        cmap=aebd_cmap_blue,
        norm=norm,
        extent=extent,
        origin='upper',
        interpolation='nearest',
    )

    # layer 2: orange-brown for classes 3,5
    ax.imshow(
        data_orange,
        aspect='auto',
        cmap=aebd_cmap_orange,
        norm=norm,
        extent=extent,
        origin='upper',
        interpolation='nearest',
    )

    # vertical line at peak lat
    ax.axvline(x=center_lat, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlim(desired_min, desired_max)

    units = getattr(AEBD['particle_backscatter_coefficient_355nm'], "units", "unknown")

    info_text = f"ATLID particle_backscatter_coefficient_355nm [{units}]"
    ax.text(
        0.011, 0.97,
        info_text,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=8,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # same minimalist axes as CPR plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    return True