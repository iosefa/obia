import time
from obia.utils.tiling import create_tiled_segments


start_time = time.time()


create_tiled_segments(
    # '/Users/iosefa/repos/hawaii-landcover/puuwaawaa/data/input.tif',
    # '/Users/iosefa/repos/hawaii-landcover/puuwaawaa/output/',
    # input_mask='/Users/iosefa/repos/hawaii-landcover/puuwaawaa/data/mask.tif',
    '/Users/iosefa/trial_small/mid-elevation.tif',
    '/Users/iosefa/trial_small/',
    input_mask='/Users/iosefa/trial_small/mask.tif',
    tile_size=2000,
    buffer=50,
    sigma=0,
    crown_radius=3,
    convert2lab=False,
    slic_zero=True
)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time taken for create_tiled_segments: {elapsed_time:.2f} seconds")
