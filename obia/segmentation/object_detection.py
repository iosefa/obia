# todo: use detection module instead.


from obia.utils.training import tile_and_process
from obia.utils.utils import save_deepforest_predictions_to_gpkg
from deepforest import main
from deepforest.visualize import plot_results

import matplotlib.pyplot as plt


m = main.deepforest()
m.load_model(
    "weecology/deepforest-tree", revision="main"
)
m.to("mps")

tile_and_process(
    raster_path="/Users/iosefa/repos/obia/docs/example_data/image.tif",
    mask_path="/Users/iosefa/repos/obia/docs/example_data/mask.tif",
    boxes_gpkg_path="/Users/iosefa/repos/obia/docs/example_data/boxes.gpkg",
    output_dir="output_tiles_blur",
    apply_clahe_flag=True,
    feather_radius=0,
    blur_kernel=11,
    darken_factor=0,
    rescale=True
)

save_deepforest_predictions_to_gpkg(
    df=img,
    tile_name="img_007.jpg",
    transforms_json="output_tiles_blur/transforms.json",
    output_gpkg="detections_img_007.gpkg"
)
