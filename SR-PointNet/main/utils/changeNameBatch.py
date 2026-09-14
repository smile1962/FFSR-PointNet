# Standard library imports
import os
from datetime import datetime, timedelta

"""
This script performs batch renaming of sequentially numbered GeoTIFF files to timestamp-based filenames.
It processes multiple subdirectories, maps sample index numbers to formatted datetime strings
according to a specified start time and fixed time step, and renames each file correspondingly.
"""

# Root directory containing all result subfolders
root_dir = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_300"

# List of subdirectories to process
subfolders = [
    "D_error", "D_ori", "D_true", "D_pred",
    "U_error", "U_ori", "U_true", "U_pred"
]

# Starting timestamp for the first sample
start_time = datetime.strptime("2019-08-19_22_15", "%Y-%m-%d_%H_%M")
# Fixed time interval between consecutive samples
time_delta = timedelta(minutes=15)

# Iterate over each subfolder
for folder in subfolders:
    folder_path = os.path.join(root_dir, folder)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    # Iterate over sample files from sample1 to sample32
    for i in range(1, 33):
        old_name = os.path.join(folder_path, f"sample{i}.tif")
        if not os.path.exists(old_name):
            print(f"File not found: {old_name}")
            continue

        # Calculate corresponding timestamp for current sample index
        new_time = start_time + (i - 1) * time_delta
        new_name_str = new_time.strftime("%Y-%m-%d_%H_%M") + ".tif"
        new_name = os.path.join(folder_path, new_name_str)

        # Execute file rename operation
        os.rename(old_name, new_name)
        print(f"Renamed {old_name} -> {new_name}")

print("All files renamed successfully.")