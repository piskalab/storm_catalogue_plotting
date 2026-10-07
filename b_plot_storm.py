import os
import gc
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
from utils.cpr_utils import plot_reflectivity, plot_doppler, plot_doppler_sw
from utils.atl_utils import plot_atlid, plot_atlid_ebd
from utils.msi_utils import plot_msi, plot_summary_horizontal
from utils.actc_utils import plot_actc
from utils.lightning_utils import get_lightning_counts, plot_lightning_info
#from config import *

def _add_inset_colorbar(fig, ax, im, color='black'):
    """Utility to add a horizontal inset colorbar in top-right corner."""
    cax = inset_axes(ax,
                     width="30%",   # fraction of parent axes
                     height="3%",   # fraction of parent axes
                     loc="upper right",
                     borderpad=0.8)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7, direction="in", length=4, color=color, labelcolor=color)
    cbar.outline.set_edgecolor(color)


def plot_storm_subplots(row, file_lists, save_dir=None, time_threshold=150, distance_threshold=2.5):
    #fig = plt.figure(figsize=(9, 9), constrained_layout=True)
    #fig = plt.figure(figsize=(6, 10.5), constrained_layout=True)
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)

    # 4 rows, 2 columns (left: plots, right: map summary)
    #gs = GridSpec(nrows=6, ncols=2, width_ratios=[6.5, 2.5], figure=fig)
    #gs = GridSpec(nrows=6, ncols=1, figure=fig)
    gs = GridSpec(nrows=3, ncols=2, figure=fig)

    # Left column axes
    ax_light   = fig.add_subplot(gs[0, 0])
    ax_sum     = fig.add_subplot(gs[1, 0])
    ax_msi     = fig.add_subplot(gs[2, 0])
    ax_refl    = fig.add_subplot(gs[0, 1])
    ax_dopp    = fig.add_subplot(gs[1, 1])
    #ax_dopp_sw = fig.add_subplot(gs[5, 0])
    ax_atl     = fig.add_subplot(gs[2, 1])
    #ax_actc    = fig.add_subplot(gs[7, 0])
    #ax_aebd     = fig.add_subplot(gs[4, 0])

    # Right column axis, spanning rows 1 & 2 (i.e., index 1 and 2)
    #ax_summary = fig.add_subplot(gs[2:4, 1], projection=ccrs.Mercator())

    # Plot each subplot
    plot0 = plot_summary_horizontal(row, ax_sum, file_lists['msi'], file_lists['li'], time_s=150)
    bt_mappable = plot_msi(row, ax_msi, file_lists['msi'])

    plot_lats, plot_counts, plot_counts_cluster, desired_min, desired_max = get_lightning_counts(
        row, file_lists['track'])

    plot2 = plot_lightning_info(row, ax_light, plot_lats, plot_counts, plot_counts_cluster, 
                                   desired_min, desired_max,
                                   time_s=time_threshold, distance_km=distance_threshold)
    plot3 = plot_reflectivity(row, ax_refl, file_lists['fmr'])
    plot4 = plot_doppler(row, ax_dopp, file_lists['cd'])
    #plot5 = plot_doppler_sw(row, ax_dopp_sw, file_lists['cd'])
    plot6 = plot_atlid(row, ax_atl, file_lists['atl'])
    #plot7 = plot_actc(row, ax_actc, file_lists['actc'])
    #plot8 = plot_atlid_ebd(row, ax_aebd, file_lists['aebd'])
    #plot9 = plot_summary(row, ax_summary, file_lists['msi'], file_lists['li'], file_lists['fmr'], time_s=150)

    if not (plot2 and plot3):
        print(f"Skipping due to missing data for orbit: {row['earthcare_id']}")
        plt.close(fig)
        return

    # --- Add inset colorbars ---
    if bt_mappable is not None:
        _add_inset_colorbar(fig, ax_msi, bt_mappable)
    if ax_refl.images:
        _add_inset_colorbar(fig, ax_refl, ax_refl.images[-1])
    if ax_dopp.images:
        _add_inset_colorbar(fig, ax_dopp, ax_dopp.images[-1])
    #if ax_dopp_sw.images:
    #    _add_inset_colorbar(fig, ax_dopp_sw, ax_dopp_sw.images[-1])
    if ax_atl.images:
        _add_inset_colorbar(fig, ax_atl, ax_atl.images[-1], color='white')

    if save_dir:
        #filename = f"EC_{row['orbit_frame']}_{row['source']}_subcluster_{row['cluster_id']}.png"
        filename = f"{row['unique_id']}.png"
        fig.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close(fig)
    else:
        plt.show()
    gc.collect()