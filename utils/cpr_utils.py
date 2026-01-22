import os
import xarray as xr
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def define_cpr_cmap():
    import matplotlib.colors as mcolors
    colors = [
        (0.0, (0.9, 0.9, 0.9)), (0.1, 'blue'), (0.2, 'blue'),
        (0.35, 'green'), (0.5, 'yellow'), (0.67, 'yellow'),
        (0.9, 'red'), (0.95, 'red'), (1.0, 'black')
    ]
    return mcolors.LinearSegmentedColormap.from_list('cpr_cmap', colors)

cpr_cmap = define_cpr_cmap()

def _load_window(ds, var_name, peak_lat, half_width=1.5):
    lat = ds['latitude'].astype('float64').values
    desired_min = peak_lat - half_width
    desired_max = peak_lat + half_width

    # mask & indices
    mask = (lat >= desired_min) & (lat <= desired_max)
    if not np.any(mask):
        return None, None, None, None, None

    i0, i1 = np.where(mask)[0][[0, -1]]
    i1 += 1  # include last index

    lat_dim = ds['latitude'].dims[0]
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

def plot_reflectivity(row, ax, ds):
    orbit = row['orbit_frame']
    peak_lat = row['peak_lat']

    out = _load_window(ds, 'reflectivity_no_attenuation_correction', peak_lat)
    if out[0] is None:
        return False
    data, center_lat, extent, desired_min, desired_max = out

    vmin, vmax = -40, 20
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    ax.imshow(data, aspect='auto', cmap=cpr_cmap, norm=norm, extent=extent, origin='upper', interpolation='none')

    # draw vertical line at peak_lat (lat units)
    ax.axvline(x=center_lat, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlim(desired_min, desired_max)

    info_text = "C-FMR Radar reflectivity [dBZ]"
    ax.text(0.011, 0.97, info_text, transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1)
    return True

def plot_doppler(row, ax, ds):
    orbit = row['orbit_frame']
    peak_lat = row['peak_lat']

    out = _load_window(ds, 'doppler_velocity_best_estimate', peak_lat)
    #out = _load_window(ds, 'doppler_velocity_integrated', peak_lat)
    if out[0] is None:
        return False
    data, center_lat, extent, desired_min, desired_max = out

    vmin, vmax = -5, 5
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    ax.imshow(data, aspect='auto', cmap='seismic', norm=norm, extent=extent, origin='upper', interpolation='none')

    ax.axvline(x=center_lat, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlim(desired_min, desired_max)

    info_text = "C-CD Doppler velocity [m/s]"
    ax.text(0.011, 0.97, info_text, transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1)
    return True

def plot_doppler_sw(row, ax, ds):
    orbit = row['orbit_frame']
    peak_lat = row['peak_lat']

    #out = _load_window(ds, 'spectrum_width_uncorrected', peak_lat)
    out = _load_window(ds, 'spectrum_width_integrated', peak_lat)
    if out[0] is None:
        return False
    data, center_lat, extent, desired_min, desired_max = out

    vmin, vmax = 0.5, 3.6
    #norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    ax.imshow(data, aspect='auto', cmap='gist_stern_r', extent=extent, origin='upper', vmin=vmin, vmax=vmax, interpolation='none')

    ax.axvline(x=center_lat, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlim(desired_min, desired_max)

    info_text = "C-CD Spectrum width [m/s]"
    ax.text(0.011, 0.97, info_text, transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1)
    return True