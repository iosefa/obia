import numpy as np
from PIL.Image import fromarray
import rasterio

from obia.utils.image import apply_clahe, apply_histogram_equalization, rescale_to_8bit


class Image:
    """
        Image
        A class for handling geographic raster image data, providing utilities to manage its properties and to convert it into visualization formats.

        Attributes
        ----------
        img_data : numpy.ndarray
            The image data array.
        crs : str or dict
            Coordinate reference system of the image.
        transform : affine.Affine
            Affine transformation matrix.
        affine_transformation : affine.Affine
            Deprecated attribute for affine transformation.
        rasterio_obj : rasterio.io.DatasetReader
            Rasterio dataset object for the image.
        valid_mask : numpy.ndarray or None
            Valid source pixels in loaded band order, shaped (height, width, bands).
            Feature extraction excludes pixels marked False by raster NoData or masks.

        Methods
        -------
        __init__(self, img_data, crs, affine_transformation, transform, rasterio_obj)
            Constructs all the necessary attributes for the Image object.
        to_image(self, bands)
            Converts specified bands of the image into an RGB image.
    """
    img_data = None
    crs = None
    transform = None
    affine_transformation = None
    rasterio_obj = None

    def __init__(self, img_data, crs, affine_transformation, transform, rasterio_obj, valid_mask=None):
        self.img_data = img_data
        self.crs = crs
        self.affine_transformation = affine_transformation
        self.transform = transform
        self.rasterio_obj = rasterio_obj
        self.valid_mask = valid_mask

    def to_image(self, bands, p_min=2, p_max=98, stretch_type=None):
        """
        always applies a linear stretch
        :param bands: A list or tuple of three integers representing the indices of the bands to use for the RGB image.
        :param p_min: Minimum percentile for rescaling.
        :param p_max: Maximum percentile for rescaling.
        :param stretch_type: Type of contrast stretching to apply ("linear", "histogram_equalization", "clahe").
        :return: A rescaled and contrast-stretched RGB image.
        """
        if not isinstance(bands, (list, tuple)) or len(bands) != 3:
            raise ValueError("'bands' should be a list or tuple of exactly three elements")

        num_bands = self.img_data.shape[2]
        rgb_data = np.empty((self.img_data.shape[0], self.img_data.shape[1], 3), dtype=np.float32)

        for i, band in enumerate(bands):
            if band >= num_bands or band < 0:
                raise IndexError(f"Band index {band} out of range. Available bands indices: 0 to {num_bands - 1}.")
            rgb_data[:, :, i] = self.img_data[:, :, band]

        rgb_data_rescaled = rescale_to_8bit(rgb_data, min=p_min, max=p_max)

        if stretch_type == "histogram_equalization":
            rgb_data_rescaled = apply_histogram_equalization(rgb_data_rescaled)
        elif stretch_type == "clahe":
            rgb_data_rescaled = apply_clahe(rgb_data_rescaled)
        elif stretch_type != None:
            raise ValueError(f"Unknown stretch_type: {stretch_type}")

        return fromarray(rgb_data_rescaled.astype(np.uint8))


def open_geotiff(image_path, bands=None):
    """
    :param image_path: Path to the GeoTIFF file to be opened.
    :type image_path: str
    :param bands: List of band indices to be read from the GeoTIFF file. If None, all bands are read.
    :type bands: list of int, optional
    :return: An Image object containing the raster data, coordinate reference system, affine transformation matrix, and the original rasterio object.
    :rtype: Image
    """
    rasterio_obj = rasterio.open(image_path)

    crs = rasterio_obj.crs
    transform = rasterio_obj.transform
    affine_transformation = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]

    x_size = rasterio_obj.width
    y_size = rasterio_obj.height
    num_bands = rasterio_obj.count

    if bands is None:
        bands = list(range(1, num_bands + 1))

    data = np.empty((y_size, x_size, len(bands)), dtype=np.float32)
    valid_mask = np.empty(data.shape, dtype=bool)

    for i, b in enumerate(bands):
        band_array = rasterio_obj.read(b, masked=True)
        data[:, :, i] = band_array.data
        valid_mask[:, :, i] = ~np.ma.getmaskarray(band_array)

    return Image(data, crs, affine_transformation, transform, rasterio_obj, valid_mask)


def _write_geotiff(pil_image, output_path, crs, transform):
    """
    Writes the given PIL image as a GeoTIFF file to the specified output path. This function
    handles both single-band and multi-band image data. It transforms the PIL image data into
    a format suitable for raster processing and writes it using the GeoTIFF driver in
    `rasterio`. The Coordinate Reference System (CRS) and affine transform associated with
    the image must be provided to correctly georeference the output GeoTIFF.

    :param pil_image: The PIL image to be written as a GeoTIFF. It can be a single-band or
        multi-band image.
    :type pil_image: PIL.Image.Image
    :param output_path: The file path where the GeoTIFF will be saved.
    :type output_path: str
    :param crs: Coordinate Reference System for the output GeoTIFF. This should be provided
        in a format understood by `rasterio`, such as an EPSG code.
    :type crs: any
    :param transform: The affine transform to be applied to the image. This should align the
        pixel grid with the georeferenced coordinate system.
    :type transform: Affine
    :return: None
    """
    data = np.array(pil_image)
    data = data.astype(np.uint8)

    bands, height, width = data.shape if len(data.shape) == 3 else (1, *data.shape)

    new_image = rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform)
    if bands > 1:
        for i in range(bands):
            new_image.write(data[i], i + 1)
    else:
        new_image.write(data, 1)
    new_image.close()
    print(f"Done Writing GeoTIFF at {output_path}")


def open_binary_geotiff_as_mask(mask_path):
    """
    :param mask_path: Path to the binary GeoTIFF file to be opened.
    :return: A tuple containing the binary mask array, bounding box, affine transform, and profile of the raster.
    """
    with rasterio.open(mask_path) as src:
        mask_array = src.read(1).astype(bool)  # Read the mask as a binary array
        transform = src.transform  # Get the affine transform of the raster
        width, height = src.width, src.height  # Get the dimensions of the raster
        profile = src.profile  # Capture the profile to use later

        # Calculate the bounding box from the transform and dimensions
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        bbox = (left, bottom, right, top)

    return mask_array, bbox, transform, profile
