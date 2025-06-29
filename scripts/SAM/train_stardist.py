from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams["image.interpolation"] = 'none'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

np.random.seed(42)
lbl_cmap = random_label_cmap()



def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y


X = sorted(glob('train/x/*.tif'))
Y = sorted(glob('train/y/*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

X = list(map(imread,X))
Y = list(map(imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

axis_norm = (0,1)   # normalize channels independently

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]


assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

def plot_img_label(img, lbl, img_title="image", lbl_title="label",
                   fname=None,  # ← NEW: pass a filename to save
                   **kwargs):
    fig, (ai, al) = plt.subplots(1, 2, figsize=(12, 5),
                                 gridspec_kw=dict(width_ratios=(1.25, 1)))
    im = ai.imshow(img, cmap="gray", vmin=0, vmax=1)
    ai.set_title(img_title)
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()

    if fname:                     # save if a file name was supplied
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)            # free memory / avoid popup windows
    return fig

i = min(9, len(X)-1)
img, lbl = X[i], Y[i]
assert img.ndim in (2,3)
img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
plot_img_label(img,lbl, "plt0.png")


# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = False and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel
)

model = StarDist2D(conf, name='stardist', basedir='models')

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")


model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)

model.optimize_thresholds(X_val, Y_val)

Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_val)]

# --- make & save the two figures ---

plot_img_label(
    X_val[0],Y_val[0], lbl_title="label GT", fname="plt3.png"
)
plot_img_label(
    X_val[0],Y_val_pred[0], lbl_title="label Pred", fname="plt4.png"
)

# ---------------------------------------------------------------------
# Save Y_val_pred[0] as GeoTIFF with the same geotransform / CRS as X_val[0]
# ---------------------------------------------------------------------
import rasterio
from rasterio import shutil as rio_shutil   # ensures creation on Windows, too

# path of the validation tile we just predicted on
x_paths = sorted(glob('train/x/*.tif'))
val_path = x_paths[ind_val[0]]              # same index as X_val[0]

out_tif = "Y_val_pred0.tif"

with rasterio.open(val_path) as src:
    meta = src.meta.copy()
    meta.update(count=1, dtype='uint16', compress='lzw')

    with rasterio.open(out_tif, 'w', **meta) as dst:
        dst.write(Y_val_pred[0].astype('uint16'), 1)

print(f"✓ wrote {out_tif} with spatial metadata copied from {val_path}")