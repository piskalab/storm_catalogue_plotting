import matplotlib
matplotlib.use("Agg")

import pandas as pd
import gc

from a_load_catalogue import (
    load_filtered_storms,
    build_file_lists,
)
from b_plot_storm import plot_storm_subplots
from config import OUTPUT_DIR

# ----------------------------
# Load storms
# ----------------------------
storm_df = load_filtered_storms()

# Optional resume
RESUME_FROM = '00998A'

resumed_df = storm_df[
    (storm_df['earthcare_id'] >= RESUME_FROM)
].reset_index(drop=True)

# Counting thresholds
time_threshold = 150
distance_threshold = 2.5

# ----------------------------
# Processing loop
# ----------------------------
for idx, row in resumed_df.iterrows():
    orbit_frame = row['earthcare_id']
    source = row['source']
    print(f"Processing storm {idx + 1}/{len(resumed_df)}: Orbit {orbit_frame}")

    if pd.isna(row.get('peak_datetime')):
        print(f"[WARN] Skipping orbit {orbit_frame} due to missing datetime")
        continue

    file_lists = build_file_lists(orbit_frame, source)

    # Proceed with your existing plotting logic
    plot_storm_subplots(
        row,
        file_lists,
        save_dir=OUTPUT_DIR,
        time_threshold=time_threshold,
        distance_threshold=distance_threshold
    )

    gc.collect()