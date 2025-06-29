# from pyforestscan.handlers import read_lidar
# from pyforestscan.visualize import plot_2d
#
# las = "/Users/iosefa/trial_small/individual_crowns_points/crown_0.las"
# las_srs = "EPSG:32605"
#
# pointclouds = read_lidar(las, las_srs, hag=True) #, bounds)
# plot_2d(
#     pointclouds[0],
#     x_dim='X', y_dim='HeightAboveGround',
#     alpha=0.5, point_size=50,
#     fig_size=(10, 10), fig_title="Crown 0"
# )

import os
from pyforestscan.handlers import read_lidar
from pyforestscan.visualize import plot_2d
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Define directories
# ----------------------------------------------------------------------
input_dir = "/Users/iosefa/trial_small/individual_crowns_points"  # Where .las files are
output_dir = "/Users/iosefa/trial_small/crown_plots"             # Where .png plots will go
os.makedirs(output_dir, exist_ok=True)

las_srs = "EPSG:32605"  # Coordinate reference system of the LAS files

# ----------------------------------------------------------------------
# 2. Iterate over all LAS files
# ----------------------------------------------------------------------
for filename in sorted(os.listdir(input_dir)):
    if filename.lower().endswith(".las"):
        las_path = os.path.join(input_dir, filename)

        # Optional: parse out the crown ID from the filename, e.g. "crown_0.las" -> 0
        # Or just use the whole filename for labeling
        crown_id = os.path.splitext(filename)[0]  # "crown_0" for example

        # ------------------------------------------------------------------
        # 3. Read the LAS file with pyforestscan
        # ------------------------------------------------------------------
        pointclouds = read_lidar(las_path, las_srs, hag=True)
        # read_lidar returns a list of DataFrames, typically one per LAS if not tiled.
        # We'll assume there's just one item:
        pc = pointclouds[0]

        # ------------------------------------------------------------------
        # 4. Plot with plot_2d
        #    We'll do X vs. HeightAboveGround, but you can choose other dims.
        # ------------------------------------------------------------------
        plot_2d(
            pc,
            x_dim="X",
            y_dim="HeightAboveGround",
            alpha=0.5,
            point_size=50,
            fig_size=(10, 10),
            fig_title=f"{crown_id}"
        )

        # ------------------------------------------------------------------
        # 5. Save the figure, then close it
        # ------------------------------------------------------------------
        out_plot = os.path.join(output_dir, f"{crown_id}.png")
        plt.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

        print(f"Saved {out_plot}")