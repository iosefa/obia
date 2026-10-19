# Chesapeake example data

These crops pair USDA NAIP 2013 imagery with Chesapeake Conservancy land-cover
reference maps from the [Chesapeake Land Cover dataset on LILA BC](https://lila.science/datasets/chesapeakelandcover/).
Credit USDA for imagery and Chesapeake Conservancy for reference labels.

Each crop is 1024 × 1024 pixels at the original 1 m resolution in EPSG:26918.
The four imagery bands retain their original order (OBIA indices 0, 1, 2, 3).
Reference codes are 1: water, 2: tree canopy/forest, 3: low vegetation/field,
4: barren land, 5: impervious (other), 6: impervious (road), and 15: NoData.
Pixel values are unchanged; reference files explicitly mark 15 as NoData.

`train_*` and `test_*` come from separate upstream training and test tiles.
Windows were selected for class diversity, without evaluating a model.
They are teaching samples, not representative accuracy samples or the complete
benchmark. The reference is an existing map, not independent field truth.

## Attribution and terms

The publisher identifies NAIP imagery as public domain. Its label-data terms
require source credit when transferring, modifying, or analysing the data,
disclaim warranties and possible errors, and place responsibility for suitable
use on the user. See the publisher's [licensing notice](https://lila.science/datasets/chesapeakelandcover/#:~:text=Licensing).
OBIA's MIT license covers its code; these data retain their source terms.

Cite: Robinson, C., Hou, L., Malkin, K., Soobitsky, R., Czawlytko, J., Dilkina, B.,
and Jojic, N. (2019). *Large Scale High-Resolution Land Cover Mapping with
Multi-Resolution Data*. CVPR. [doi:10.1109/CVPR.2019.01301](https://doi.org/10.1109/CVPR.2019.01301).

## Reproduce the crops

`metadata.json` records source URLs, exact pixel windows, grid metadata, and
SHA-256 hashes of both files and uncompressed pixel arrays. From the repository:

```bash
python scripts/prepare_chesapeake_sample.py --output-dir /tmp/chesapeake-sample
```

This fetches only the required GeoTIFF windows. The script verifies pixel hashes
against the manifest. Compressed file hashes can differ with GDAL versions;
the uncompressed pixels and georeferencing must match.
