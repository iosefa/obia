"""
step2.py – tile → fine-tune DeepForest → predict

Changes vs. previous version
----------------------------
1. Wrapped everything in `main()` and guarded with
   `if __name__ == "__main__": …`  (no recursive spawn).
2. Force DataLoader to single-process with `m.config["workers"] = 0`.
3. Added a tiny helper to bump Albumentations warning.
"""

import os, glob, json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from deepforest import main as df

from obia.utils.training import tile_and_process
from obia.utils.utils import save_deepforest_predictions_to_gpkg


# ─── PATHS ──────────────────────────────────────────────────────────
TRAIN_RASTER   = "/Users/iosefa/repos/obia/docs/example_data/train.tif"
TRAIN_POLYGONS = "/Users/iosefa/repos/obia/docs/example_data/b.gpkg"
TRAIN_DIR      = "/Users/iosefa/repos/obia/docs/example_data/training_tiles"

BIG_RASTER     = "/Users/iosefa/repos/obia/docs/example_data/image_full.tif"
BIG_MASK       = "/Users/iosefa/repos/obia/docs/example_data/mask_full.tif"
BIG_TILES_DIR  = "/Users/iosefa/repos/obia/docs/example_data/output_tiles_full"
DETECTIONS_DIR = "/Users/iosefa/repos/obia/docs/example_data/detections"

MODEL_PATH     = "/Users/iosefa/repos/obia/docs/example_data/deepforest_finetuned.pt"

TILE_SIZE_M  = 179.2    # 512 px @ 0.35 m
OVERLAP_M    = 20.0
EPOCHS       = 5
BATCH_SIZE   = 2


# ─── HELPERS ────────────────────────────────────────────────────────
def annotations_to_csv(ann_json: Path, root_dir: Path, csv_out: Path):
    with open(ann_json) as f:
        ann = json.load(f)

    rows = []
    for item in ann.values():
        img = root_dir / item["file_name"]
        for xmin, ymin, xmax, ymax in item["boxes"]:
            rows.append(
                dict(
                    image_path=str(img),
                    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                    label="Tree",
                )
            )
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"[✓] wrote {len(rows):,} boxes → {csv_out}")


# ─── MAIN PIPELINE ─────────────────────────────────────────────────
def main():
    # 1. ─ tile & CSV for training (runs once) ──────────────────────
    train_dir = Path(TRAIN_DIR)
    train_dir.mkdir(parents=True, exist_ok=True)
    train_csv = train_dir / "train.csv"

    if not train_csv.exists():
        print("• tiling training AOI …")
        tile_and_process(
            raster_path=TRAIN_RASTER,
            mask_path=None,
            boxes_gpkg_path=TRAIN_POLYGONS,
            output_dir=str(train_dir),
            tile_size=TILE_SIZE_M,
            overlap=0.0,
            selected_bands=(4, 2, 1),
            apply_clahe_flag=False,
            rescale=True,
            blur_kernel=0,
            darken_factor=1.0,
        )
        annotations_to_csv(train_dir / "annotations.json", train_dir, train_csv)
    else:
        print(f"• using cached training CSV → {train_csv}")

    # 2. ─ fine-tune DeepForest (once) ──────────────────────────────
    if not Path(MODEL_PATH).exists():
        print("• fine-tuning …")
        m = df.deepforest(num_classes=1)
        m.load_model("weecology/deepforest-tree", revision="main")

        m.config["train"]["csv_file"]   = str(train_csv)
        m.config["train"]["root_dir"]   = str(train_dir)
        m.config["validation"]["csv_file"] = None
        m.config["batch_size"] = BATCH_SIZE
        m.config["n_epochs"]   = EPOCHS
        m.config["workers"]    = 0          # <── single-proc loader

        m.create_trainer()
        m.trainer.fit(m)
        m.save_model(MODEL_PATH)
        print(f"[✓] model saved → {MODEL_PATH}")
    else:
        print("• loading cached model …")
        m = df.deepforest()
        m.load_model(MODEL_PATH)

    m.to("mps")

    # 3. ─ tile big raster for prediction ───────────────────────────
    print("• tiling big raster …")
    tile_and_process(
        raster_path=BIG_RASTER,
        mask_path=BIG_MASK,
        boxes_gpkg_path=None,
        output_dir=BIG_TILES_DIR,
        tile_size=TILE_SIZE_M,
        overlap=OVERLAP_M,
        selected_bands=(4, 2, 1),
        apply_clahe_flag=False,
        rescale=True,
        blur_kernel=3,
        darken_factor=0.85,
    )

    # 4. ─ inference per tile ───────────────────────────────────────
    Path(DETECTIONS_DIR).mkdir(parents=True, exist_ok=True)
    tjson = Path(BIG_TILES_DIR) / "transforms.json"

    from tqdm.auto import tqdm
    print("• predicting …")
    for img in tqdm(sorted(glob.glob(f"{BIG_TILES_DIR}/*.jpg"))):
        name = Path(img).stem
        try:
            preds = m.predict_image(path=img)
            if preds.empty:
                continue
            save_deepforest_predictions_to_gpkg(
                df=preds,
                tile_name=f"{name}.jpg",
                transforms_json=str(tjson),
                output_gpkg=f"{DETECTIONS_DIR}/{name}.gpkg",
            )
        except Exception as e:
            print(f"[warn] {name}: {e}")

    print("🎉  done – detections in", DETECTIONS_DIR)


# ─── RUN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

# # import glob
# #
# # from obia.utils.training import tile_and_process
# # from obia.utils.utils import save_deepforest_predictions_to_gpkg
# # from deepforest import main
# # from deepforest.visualize import plot_results
# #
# # import matplotlib.pyplot as plt
# #
# #
# # m = main.deepforest()
# # m.load_model(
# #     "weecology/deepforest-tree", revision="main"
# # )
# # m.to("mps")
# #
# # tile_size = 179.2 * 1
# #
# # tile_and_process(
# #     raster_path="/Users/iosefa/repos/obia/docs/example_data/image+chm_full.tif",
# #     mask_path="/Users/iosefa/repos/obia/docs/example_data/mask_full.tif",
# #     boxes_gpkg_path=None,
# #     output_dir="/Users/iosefa/repos/obia/docs/example_data/output_tiles_full",
# #     tile_size=tile_size, # 179.2 is 512 px
# #     overlap=20.0,
# #     selected_bands=(4,2,1),
# #     apply_clahe_flag=False,
# #     feather_radius=0,
# #     blur_kernel=3,
# #     darken_factor=0.85,
# #     rescale=True
# # )
# #
# # for image_path in glob.glob("/Users/iosefa/repos/obia/docs/example_data/output_tiles_full/*.jpg"):
# #     image_name = # get image name
# #     img = m.predict_image(path=image_path)
# #     save_deepforest_predictions_to_gpkg(
# #         df=img,
# #         tile_name="img_3743",
# #         transforms_json="../example_data/output_tiles_full/transforms.json",
# #         output_gpkg=f"detections/{image_name.gpkg}.gpkg"
# #     )
#
# import os
# import glob
# from obia.utils.training import tile_and_process
# from obia.utils.utils import save_deepforest_predictions_to_gpkg
# from deepforest import main
# from deepforest.visualize import plot_results
# import matplotlib.pyplot as plt
#
#
#
# tile_size = 179.2 * 1   # 512 px at 0.35 m
#
# tile_and_process(
#     raster_path="/Users/iosefa/repos/obia/docs/example_data/image_full.tif",
#     mask_path="/Users/iosefa/repos/obia/docs/example_data/mask_full.tif",
#     boxes_gpkg_path=None,
#     output_dir="/Users/iosefa/repos/obia/docs/example_data/output_tiles_full",
#     tile_size=tile_size,
#     overlap=20.0,
#     selected_bands=(4, 2, 1),
#     apply_clahe_flag=False,
#     feather_radius=0,
#     blur_kernel=3,
#     darken_factor=0.85,
#     rescale=True,
# )
#
# m = main.deepforest()
# m.load_model("weecology/deepforest-tree", revision="main")
# m.to("mps")
#
# # Iterate over the generated tiles
# for image_path in glob.glob(
#     "/Users/iosefa/repos/obia/docs/example_data/output_tiles_full/*.jpg"
# ):
#     # → e.g. image_path == “…/output_tiles_full/img_3743.jpg”
#     try:
#         image_name = os.path.splitext(os.path.basename(image_path))[0]  # img_3743
#
#         # Run DeepForest inference
#         preds = m.predict_image(
#             path=image_path
#         )
#
#         # Save predictions to a GeoPackage named after the tile
#         save_deepforest_predictions_to_gpkg(
#             df=preds,
#             tile_name=image_name+".jpg",  # use the same name inside the GPKG
#             transforms_json="/Users/iosefa/repos/obia/docs/example_data/output_tiles_full/transforms.json",
#             output_gpkg=f"/Users/iosefa/repos/obia/docs/example_data/detections/{image_name}.gpkg"
#         )
#     except Exception as e:
#         pass
