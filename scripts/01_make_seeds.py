import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from obia.utils.seeds import make_canonical_seeds
from obia.utils.cost import make_cost_surface


make_cost_surface(
    "/Users/iosefa/repos/obia/docs/example_data/site_2/image.tif",
    "/Users/iosefa/repos/obia/docs/example_data/site_2/chm.tif",
    "/Users/iosefa/repos/obia/scripts/cost_full.tif"
)

make_canonical_seeds(
    "/Users/iosefa/repos/obia/scripts/CHM_Seeds/chm_seeds.gpkg",
    "/Users/iosefa/repos/obia/scripts/FirstReturn_Seeds/den_seeds.gpkg",
    "/Users/iosefa/repos/obia/docs/example_data/site_2/chm.tif",
    "/Users/iosefa/repos/obia/scripts/cost_full.tif",
    "/Users/iosefa/repos/obia/scripts/seeds.gpkg",
)
