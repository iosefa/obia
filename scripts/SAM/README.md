
Data Preparation:

1. create CHM
2. create Mosaic RNN
3. create Density tiff
4. create cost raster

python scripts/final/002_make_cost_surface.py --wv3  /Users/iosefa/repos/obia/docs/example_data/full/image_full.tif  --chm  /Users/iosefa/repos/obia/docs/example_data/full/chm_full.tif  --out  /Users/iosefa/repos/obia/scripts/final/tile_cost.tif


5. create canonical seeds

python scripts/final/003_make_canonical_seeds.py \
  --chm-seeds  /Users/iosefa/repos/obia/scripts/final/chm_seeds.gpkg \
  --den-seeds  /Users/iosefa/repos/obia/scripts/final/den_seeds.gpkg \
  --chm-raster /Users/iosefa/repos/obia/docs/example_data/full/chm_full.tif \
  --cost-surface /Users/iosefa/repos/obia/scripts/final/cost.tif \
  --out        /Users/iosefa/repos/obia/scripts/final/seeds.gpkg \
  --eps-scale 0.4 --min-eps 2 --max-eps 8 --min-samples 2 --z-thresh -1 \
  --keep-all-stage1            \
  --max-per-cluster 0          \
  --nms-base 0 --nms-scale 0   \
  --merge-radius 1.5           \
  --cost-weight 0.5            \
  --xy-thresh 0.8              \
  --dz-merge 0                 \
  --debug-dist


To create the SAM model:

STEP 01:
01_prepare_chips
    _This script will prepare the data to train a SAM model. It requires an image (8-band multispectral) and a vector of crowns. It will create native np array files that will be used to train a custom SAM model._
    - inputs: 
        - image: input image. 3-band (7,5,3)
        - crown_mask: gpkg of training crowns
    - outputs:
        - writes npy files that contain training image tiles and masks

STEP 02:
02_train_sam_decoder
    _This script trains a custom SAM model_
    - inputs:
        - training chips (output from step 1 above)
        - SAM vit pth file with large model weights
    - outputs:
        - the newly trained model weights.

The above steps only need to be done once. Then the model should know!


STEP 1:
03_predict_all
    - inputs:
        - seeds
        - image
        - chm
        - SAM base model
        - SAM custom model



This leaves us with individual tiles of probability. 

make_stardist_tiles --> composite_tiles_ns 
merge_prob_tiles.py: takes composite_tiles_ns --> canopy_index.tif


