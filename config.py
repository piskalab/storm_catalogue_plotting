import os
from pathlib import Path

# ----------------------------
# Local paths
# ----------------------------
LI_BASE = '/home/bpiskala/Object_Data'
JSON_PATH  = '/home/bpiskala/AeroStorm/lightning2ec_maap/lightning2earthcare/storm_catalogue_202408.json'

LI_PATH  = os.path.join(LI_BASE, 'lightning_processing/lightning_groups')
TRACK_COUNTS_PATH = os.path.join(LI_BASE, 'lightning_processing/track_counts')

OUTPUT_DIR = './output/'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ----------------------------
# MAAP / STAC catalogue config
# ----------------------------
CATALOG_URL = 'https://catalog.maap.eo.esa.int/catalogue/'
EC_COLLECTION = ['EarthCAREL2Validated_MAAP', 'EarthCAREL1Validated_MAAP']

# I/O tuning for remote reads
IO_PARAMS = {
    "fsspec_params": {
        "cache_type": "blockcache",
        "block_size": 8 * 1024 * 1024,
    },
    "h5py_params": {
        "driver_kwds": {
            "rdcc_nbytes": 8 * 1024 * 1024,
        }
    }
}

# Product short → STAC productType
PRODUCT_MAP = {
    'fmr': 'CPR_FMR_2A',
    'cd':  'CPR_CD__2A',
    'msi': 'MSI_RGR_1C',
    'ctc': 'CPR_TC__2A',
    'actc': 'AC__TC__2B',
    'atl': 'ATL_NOM_1B',
    'aebd': 'ATL_EBD_2A',
}