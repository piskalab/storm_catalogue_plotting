import pandas as pd
import geopandas as gpd
import fsspec
import xarray as xr
from pystac_client import Client

from config import (
    CATALOG_URL, EC_COLLECTION, IO_PARAMS, PRODUCT_MAP,
    BUCKET, PREFIX, S3_STORAGE_OPTIONS, STORM_FILE,
    EARTHCARE_ID_MAPPING_FILE, TRACK_FILE_MAP,
)
from token_handling import get_earthcare_token

# ----------------------------
# Storm list loader
# ----------------------------
def load_filtered_storms() -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(
        f"{BUCKET}{PREFIX}{STORM_FILE}",
        storage_options=S3_STORAGE_OPTIONS,
    )

    gdf["peak_datetime"] = pd.to_datetime(gdf["peak_datetime"])

    # gdf = gdf[(gdf["peak_lon"] >= 9) & (gdf["peak_lon"] <= 22) &
    #           (gdf["peak_lat"] >= 47) & (gdf["peak_lat"] <= 52)]
    # gdf = gdf[(gdf['first_lightning_min'] >= -5) & (gdf['first_lightning_min'] <= 0)]

    return gdf

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
_mapping_cache = None

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

def fetch_ec_datasets(orbit_frame: str):
    """Returns dict with keys in {'fmr','cd','msi'} → xr.Dataset. Uses your exact filter string."""
    if orbit_frame in _ds_cache:
        return _ds_cache[orbit_frame]

    orbit_num_str, frame = _parse_orbit_frame(orbit_frame)

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

def _load_earthcare_mapping() -> pd.DataFrame:
    """Load the EarthCARE ID → lightning file mapping once."""
    global _mapping_cache

    if _mapping_cache is None:
        _mapping_cache = pd.read_parquet(
            f"{BUCKET}{PREFIX}{EARTHCARE_ID_MAPPING_FILE}",
            storage_options=S3_STORAGE_OPTIONS,
        ).set_index("earthcare_id")

    return _mapping_cache

def fetch_li_datasets(orbit_frame: str, source: str):
    """Return lightning data matching this EarthCARE orbit/frame and source."""
    key = (source, orbit_frame)
    if key in _li_cache:
        return _li_cache[key]

    mapping = _load_earthcare_mapping()

    matched_files = mapping.loc[orbit_frame].values[0]

    selected_files = [
        filename
        for filename in matched_files
        if f"_{source}_" in filename
    ]

    selected_file = selected_files[0]

    gdf = gpd.read_parquet(
        f"{BUCKET}{PREFIX}{selected_file}",
        storage_options=S3_STORAGE_OPTIONS,
        filters=[("earthcare_id", "==", orbit_frame)],
    )

    _li_cache[key] = gdf
    return gdf

def fetch_track_datasets(orbit_frame: str, source: str):
    """Return track lightning data matching this EarthCARE orbit/frame."""
    key = (source, orbit_frame)
    if key in _track_cache:
        return _track_cache[key]

    selected_file = TRACK_FILE_MAP.get(source)

    gdf = gpd.read_parquet(
        f"{BUCKET}{PREFIX}{selected_file}",
        storage_options=S3_STORAGE_OPTIONS,
        filters=[("earthcare_id", "==", orbit_frame)],
    )

    _track_cache[key] = gdf
    return gdf

def build_file_lists(orbit_frame: str, source: str):
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