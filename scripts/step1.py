"""
#################################
# Step 1: Panoptic Segmentation #
#################################
The objective of this step is to segment the image into panoptic segments.
These represent individually identified and labelled trees. The rough
steps are:

1. mask any area under 3m height (using CHM model). This is done
   in a separate step but included for reference here.

2. segment using SLIC. This over-segments only pixels over 3m.
   SLIC uses multispectral bands and a CHM band.

2. detect crowns using a resnet detector. This is a multi-step
   process that includes tiling the image, integrating a CHM to
   highlight the trees, and training a CNN to detect individual
   tree crowns. Can use any object detection model. We use pre-
   trained models from deepforest and retinanet.

3. merge SLIC segments (over-segmentation) within each detected
   crown.
"""
import time
import numpy as np
import math

from obia.handlers.geotif import open_binary_geotiff_as_mask, open_geotiff
from obia.segmentation.segment_boundaries import create_segments
from obia.utils.tiling import create_tiled_segments

# IMAGE_PATH = '../docs/example_data/image+chm_full.tif'
# MASK_PATH = '../docs/example_data/mask_full.tif'
# SLIC_PATH = '../docs/example_data/slic_segments_full.tif'

IMAGE_PATH = '../docs/example_data/image_train.tif'
MASK_PATH = '../docs/example_data/mask_train.tif'
SLIC_PATH = '../docs/example_data/slic_segments_train.tif'

########################
# CREATE SLIC SEGMENTS #
########################
start_time = time.time()

# area_threshold = np.pi ** 2

create_tiled_segments(
    IMAGE_PATH,
    SLIC_PATH,
    input_mask=MASK_PATH,
    method="slic",
    segmentation_bands=[8,7,4,1],
    tile_size=800, # 200
    buffer=30, # 30
    crown_radius=2
)
#
# area_threshold = np.pi ** 2
# segments = segments[segments.geometry.area >= area_threshold]
#
# segments.to_file(SLIC_PATH)

end_time = time.time()

print(f"Time taken to create SLIC segments: {end_time - start_time:.2f} seconds")
