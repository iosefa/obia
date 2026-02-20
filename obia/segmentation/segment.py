import numpy as np
from PIL.Image import fromarray
from PIL.Image import Image as PILImage
from skimage.segmentation import mark_boundaries

from obia.segmentation.segment_boundaries import create_segments
from obia.segmentation.segment_statistics import create_objects


class Segments:
    """
    Segments class for handling and processing image segments.

    Attributes:
        _segments: Internal representation of segments data.
        segments: Public representation of segments data.
        method: Method used for segmentation.
        params: Additional parameters for segmentation.

    Methods:
        __init__(_segments, segments, method, **kwargs):
            Initializes the Segments object with segments data, a method, and additional parameters.

        to_segmented_image(image):
            Converts a given PIL Image to a segmented image with boundaries overlaid.

        write_segments(file_path):
            Writes the segments data to a file.
    """
    _segments = None
    segments = None
    method = None
    params = {}

    def __init__(self, _segments, segments, method, **kwargs):
        self._segments = _segments
        self.segments = segments
        self.method = method
        self.params.update(kwargs)

    def to_segmented_image(self, image):
        """
        :param image: An input image expected to be of type PIL Image.
        :return: A new image with the boundaries marked on the segments.
        """
        if not isinstance(image, PILImage):
            raise TypeError('Input must be a PIL Image')
        img = np.array(image)
        boundaries = mark_boundaries(img, self._segments)
        boundaries_int = boundaries * 255

        masked_img = boundaries_int.copy()
        return fromarray(masked_img.astype(np.uint8))

    def write_segments(self, file_path):
        """
        :param file_path: The path to the file where segments will be written.
        :return: None
        """
        self.segments.to_file(file_path)


def segment(image, segmentation_bands=None, statistics_bands=None,
            method="slic", calc_mean=True, calc_variance=True,
            calc_skewness=True, calc_kurtosis=True, calc_contrast=True,
            calc_dissimilarity=True, calc_homogeneity=True, calc_ASM=True,
            calc_energy=True, calc_correlation=True, **kwargs):
    """
    :param image: Input image for segmentation and feature extraction.
    :param segmentation_bands: Bands to be used for segmentation. Default is None.
    :param statistics_bands: Bands to be used for statistical calculations. Default is None.
    :param method: Segmentation method to be used. Default is "slic".
    :param calc_mean: Flag to calculate mean of segments. Default is True.
    :param calc_variance: Flag to calculate variance of segments. Default is True.
    :param calc_skewness: Flag to calculate skewness of segments. Default is True.
    :param calc_kurtosis: Flag to calculate kurtosis of segments. Default is True.
    :param calc_contrast: Flag to calculate contrast of segments. Default is True.
    :param calc_dissimilarity: Flag to calculate dissimilarity of segments. Default is True.
    :param calc_homogeneity: Flag to calculate homogeneity of segments. Default is True.
    :param calc_ASM: Flag to calculate angular second moment of segments. Default is True.
    :param calc_energy: Flag to calculate energy of segments. Default is True.
    :param calc_correlation: Flag to calculate correlation of segments. Default is True.
    :param kwargs: Additional parameters for segmentation and feature extraction.
    :return: Segments object containing the segmented image and statistics.
    """
    segments_gdf = create_segments(image, segmentation_bands=segmentation_bands, method=method, **kwargs)
    objects_gdf = create_objects(segments_gdf, image, spectral_bands=statistics_bands, calc_mean=calc_mean,
                                 calc_variance=calc_variance, calc_skewness=calc_skewness, calc_kurtosis=calc_kurtosis,
                                 calc_contrast=calc_contrast, calc_dissimilarity=calc_dissimilarity,
                                 calc_homogeneity=calc_homogeneity, calc_ASM=calc_ASM, calc_energy=calc_energy,
                                 calc_correlation=calc_correlation)

    return Segments(segments_gdf, objects_gdf, method, **kwargs)
