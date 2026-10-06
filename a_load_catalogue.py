import os
import json

import pandas as pd
import fsspec
import xarray as xr
from pystac_client import Client

from config import (
    CATALOG_URL, EC_COLLECTION, IO_PARAMS, PRODUCT_MAP,
    LI_PATH, TRACK_COUNTS_PATH
)
from token_handling import get_earthcare_token

# ----------------------------
# Storm list loader
# ----------------------------
def load_filtered_storms(json_path: str) -> pd.DataFrame:
    with open(json_path, "r") as f:
        obj = json.load(f)

    df = pd.DataFrame(obj["data"])
    df["peak_datetime"] = pd.to_datetime(df["peak_datetime"])
    df["minute_counts"] = df["minute_counts"].apply(
        lambda d: {int(k): v for k, v in d.items()}
    )
    #df = df[(df['first_lightning_min'] >= -5) & (df['first_lightning_min'] <= 0)]
    df = df[
     (df["peak_lon"] >= 9) & (df["peak_lon"] <= 22) &
     (df["peak_lat"] >= 47) & (df["peak_lat"] <= 52)
     #(df["peak_lon"] >= -75.4) & (df["peak_lon"] <= -72.4) & # colLMA
     #(df["peak_lat"] >= 6) & (df["peak_lat"] <= 8) # colLMA
     #(df["peak_lon"] >= -1) & (df["peak_lon"] <= 4) & # eLMA
     #(df["peak_lat"] >= 39.5) & (df["peak_lat"] <= 43) # eLMA
     #(df["peak_lon"] > -77.875) & (df["peak_lon"] < -73.304) & # NALMA
     #(df["peak_lat"] > 36.294) & (df["peak_lat"] < 39.891) # NALMA
]
    return df

# ----------------------------
catalog = Client.open(CATALOG_URL)
EARTHCARE_TOKEN = get_earthcare_token()
fs = fsspec.filesystem(
    "https",
    headers={"Authorization": f"Bearer {EARTHCARE_TOKEN}"},
    **IO_PARAMS["fsspec_params"]
)

# reuse already-opened datasets per orbit_frame
_ds_cache = {}
_li_cache = {}
_track_cache = {}

def _parse_orbit_frame(orbit_frame: str):
    """'07136A' -> ('07136', 'A')  (string keeps leading zeros!)"""
    return orbit_frame[:-1], orbit_frame[-1]

def _open_remote_ds(href: str) -> xr.Dataset:
    with fs.open(href, "rb") as fh:
        ds = xr.open_dataset(
            fh,
            engine="h5netcdf",
            **IO_PARAMS["h5py_params"],
            group="ScienceData",
        )
        ds.load()
    return ds

def _open_local_ds(path: str):
    # try h5netcdf first, fallback to default
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except Exception:
        return xr.open_dataset(path)

def fetch_ec_datasets(orbit_frame: str):
    """Returns dict with keys in {'fmr','cd','msi'} → xr.Dataset. Uses your exact filter string."""
    if orbit_frame in _ds_cache:
        return _ds_cache[orbit_frame]

    orbit_num_str, frame = _parse_orbit_frame(orbit_frame)

    # filter_str = (
    #     f"frame = '{frame}' and orbitNumber = {orbit_num_str} and (productVersion = 'ba' or productVersion = 'ae') and "
    #     f"(productType = '{PRODUCT_MAP['fmr']}' or "
    #     f" productType = '{PRODUCT_MAP['cd']}' or "
    #     f" productType = '{PRODUCT_MAP['msi']}' or "
    #     f" productType = '{PRODUCT_MAP['ctc']}' or "
    #     f" productType = '{PRODUCT_MAP['atl']}' or "
    #     f" productType = '{PRODUCT_MAP['actc']}')"
    # )

    filter_str = (
        f"frame = '{frame}' and orbitNumber = {orbit_num_str} and ("
        f"(productType = '{PRODUCT_MAP['fmr']}'  and (productVersion = 'ba' or productVersion = 'bc')) or "
        f"(productType = '{PRODUCT_MAP['cd']}'   and (productVersion = 'ba' or productVersion = 'bc')) or "
        f"(productType = '{PRODUCT_MAP['msi']}'  and (productVersion = 'ba' or productVersion = 'bc')) or "
        #f"(productType = '{PRODUCT_MAP['ctc']}'  and (productVersion = 'ba' or productVersion = 'bc')) or "
        #f"(productType = '{PRODUCT_MAP['aebd']}' and (productVersion = 'ba' or productVersion = 'bc')) or "
        f"(productType = '{PRODUCT_MAP['atl']}'  and (productVersion = 'ba' or productVersion = 'bc'))"
        #f"(productType = '{PRODUCT_MAP['actc']}' and (productVersion = 'ba' or productVersion = 'bc'))"
        f")"
    )

    search = catalog.search(
        collections=EC_COLLECTION,
        filter=filter_str,
        method='GET',
        max_items=10
    )
    items = list(search.items())

    prod = {}
    for item in items:
        ptype = item.properties.get("product:type", "")
        asset = item.assets.get("enclosure_h5")
        href = asset.href
        ds = _open_remote_ds(href)
        if ptype == PRODUCT_MAP['fmr']:
            prod['fmr'] = ds
        elif ptype == PRODUCT_MAP['cd']:
            prod['cd'] = ds
        elif ptype == PRODUCT_MAP['msi']:
            prod['msi'] = ds
        elif ptype == PRODUCT_MAP['ctc']:
            prod['ctc'] = ds
        elif ptype == PRODUCT_MAP['actc']:
            prod['actc'] = ds
        elif ptype == PRODUCT_MAP['atl']:
            prod['atl'] = ds
        elif ptype == PRODUCT_MAP['aebd']:
            prod['aebd'] = ds

    _ds_cache[orbit_frame] = prod
    return prod

def _pick_single_file(dir_path: str, orbit_frame: str, source: str) -> str | None:
    """Return the single matching path. If multiple, pick the newest by mtime."""
    if not os.path.isdir(dir_path):
        return None
    matches = [
        os.path.join(dir_path, fn) 
        for fn in os.listdir(dir_path) 
        if orbit_frame in fn and source in fn
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    # choose the most recently modified if more than one
    return max(matches, key=os.path.getmtime)

def fetch_li_datasets(orbit_frame: str, source: str):
    """Return a single xr.Dataset (or None) for LI matching this orbit_frame."""
    key = (source, orbit_frame)
    if key in _li_cache:
        return _li_cache[key]
    path = _pick_single_file(LI_PATH, orbit_frame, source)
    ds = _open_local_ds(path) if path else None
    _li_cache[key] = ds
    return ds

def fetch_track_datasets(orbit_frame: str, source: str):
    """Return a single xr.Dataset (or None) for TRACK COUNTS matching this orbit_frame."""
    key = (source, orbit_frame)
    if key in _track_cache:
        return _track_cache[key]
    path = _pick_single_file(TRACK_COUNTS_PATH, orbit_frame, source)
    ds = _open_local_ds(path) if path else None
    _track_cache[key] = ds
    return ds

def build_file_lists(orbit_frame: str, source: str):
    """file_lists for plot_storm_subplots: datasets only (no paths, no lists)."""
    prod = fetch_ec_datasets(orbit_frame)
    return {
        'fmr':   prod.get('fmr'),
        'cd':    prod.get('cd'),
        'msi':   prod.get('msi'),
        'ctc':   prod.get('ctc'),
        'actc':  prod.get('actc'),
        'atl':   prod.get('atl'),
        'aebd':  prod.get('aebd'),
        'li':    fetch_li_datasets(orbit_frame, source),
        'track': fetch_track_datasets(orbit_frame, source),
    }