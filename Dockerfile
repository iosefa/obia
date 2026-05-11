FROM condaforge/miniforge3:latest

LABEL org.opencontainers.image.title="OBIA"
LABEL org.opencontainers.image.description="Object-based image analysis tools for geospatial rasters"
LABEL org.opencontainers.image.source="https://github.com/iosefa/obia"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /opt/obia

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY obia ./obia

RUN conda install -y -c conda-forge \
        "python=3.11" \
        pip \
        "setuptools>=68" \
        wheel \
        "numpy>=2.1.1" \
        "rasterio>=1.3.11" \
        "shapely>=2.0.3" \
        "pillow>=10.3.0" \
        "pandas>=2.2.2" \
        "geopandas>=1.0.1" \
        "scipy>=1.14.1" \
        "scikit-image>=0.23.2" \
        "tqdm>=4.66.2" \
        "scikit-learn>=1.4.2" \
        "affine>=2.4.0" \
        "pyproj>=3.6.1" \
        "matplotlib>=3.8" \
        requests \
        opencv \
        gdal \
        pdal \
        python-pdal \
    && pip install --no-cache-dir --no-deps pyforestscan . \
    && conda clean -afy

CMD ["python"]
