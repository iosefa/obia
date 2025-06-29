import os
import glob
import time
import numpy as np
import numpy.lib.recfunctions as rfn
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for the progress bar
from pyforestscan.handlers import read_lidar, write_las

# Define the directories
laz_directory = '/Users/iosefa/laz'
output_directory = '/Users/iosefa/laz_hag'

# Get all the laz files
laz_files = glob.glob(os.path.join(laz_directory, '*.laz'))


def process_laz_file(laz_file):
    """
    Function to process a single laz file.
    """
    filename = os.path.basename(laz_file)
    output_file = os.path.join(output_directory, filename)

    # Read lidar data
    arrays = read_lidar(laz_file, "EPSG:32605", hag=True)

    # Convert 'HeightAboveGround' to float32
    pointclouds = arrays[0]
    pointclouds = rfn.rec_drop_fields(pointclouds, 'HeightAboveGround')
    height_above_ground_f4 = arrays[0]['HeightAboveGround'].astype(np.float32)
    pointclouds = rfn.rec_append_fields(pointclouds, 'HeightAboveGround', height_above_ground_f4)

    # Write processed pointclouds to output
    write_las(pointclouds, output_file, "EPSG:32605")


if __name__ == '__main__':
    # Start timing
    start_time = time.time()

    # Run the processing in parallel using ProcessPoolExecutor and tqdm
    with ProcessPoolExecutor() as executor:
        # Wrap executor.map with tqdm for progress tracking
        list(tqdm(executor.map(process_laz_file, laz_files), total=len(laz_files), desc="Processing LAZ files"))

    # End timing
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.6f} seconds")
