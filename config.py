import os
from pathlib import Path

# ----------------------------
# Local directories
# ----------------------------
LI_BASE = '/home/bpiskala/Object_Data'
JSON_PATH  = '/home/bpiskala/AeroStorm/lightning2ec_maap/lightning2earthcare/EarthCARE_lightning_storm_catalogue_FINAL.json'
#JSON_PATH = '/home/bpiskala/repositories/lightning2earthcare/EarthCARE_lightning_storm_catalogue_20260201.json'

LI_PATH  = os.path.join(LI_BASE, 'lightning_processing/lightning_groups_20240801_20260131')
#LI_PATH  = os.path.join(LI_BASE, 'lightning_processing/lightning_groups_20260201')
TRACK_COUNTS_PATH = os.path.join(LI_BASE, 'lightning_processing/track_counts_20240801_20260131')
#TRACK_COUNTS_PATH = os.path.join(LI_BASE, 'lightning_processing/track_counts_20260201')

# ----------------------------
# EarthCODE bucket
# ----------------------------
# BUCKET = "s3://EarthCODE/"
# PREFIX = "OSCAssets/storm-data/"

# ENDPOINT_URL = "https://s3.waw4-1.cloudferro.com"
# REGION_NAME = "eu-west-2"

# S3_STORAGE_OPTIONS = {
#     "anon": True,
#     "client_kwargs": {
#         "endpoint_url": ENDPOINT_URL,
#         "region_name": REGION_NAME,
#     },
# }

# ----------------------------
# Storm / lightning parquet files
# ----------------------------
# STORM_FILE = "EC_lightning_clusters.parquet"
# EARTHCARE_ID_MAPPING_FILE = "earthcare_id_mapping.parquet"

# TRACK_FILE_MAP = {
#     "GLM": "EC_track_lightning_GLM.parquet",
#     "LI": "EC_track_lightning_LI.parquet",
# }

# ----------------------------
# Output
# ----------------------------
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