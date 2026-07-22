# Map Algebra with NumPy

Inspired by the ARC/INFO GRID implementation of [Map Algebra](https://en.m.wikipedia.org/wiki/Map_algebra).

## Installation

```bash
# PyPI
pip install glidergun
```

```bash
# UV
uv add glidergun
```

For CUDA acceleration, install the PyTorch CUDA wheels before installing glidergun.
## Installation

```bash
# PyPI
pip install glidergun
```

```bash
# UV
uv add glidergun
```

For CUDA acceleration, install the PyTorch CUDA wheels before installing glidergun.

```bash
# PyPI
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# PyPI
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install glidergun
```

```bash
# UV
uv add torch torchvision --index pytorch-cu130=https://download.pytorch.org/whl/cu130
uv add glidergun
```

## Getting Started

### Terrain analysis

```python
# Downloads a Copernicus DEM and creates a hillshade GeoTIFF.
# Downloads a Copernicus DEM and creates a hillshade GeoTIFF.
from glidergun import grid

dem = grid("cop-dem-glo-90", (137.8, 34.5, 141.1, 36.8))
hillshade = dem.hillshade()

dem.save("dem.tif")
hillshade.save("hillshade.tif", "uint8")
```

### Image segmentation

```python
# Downloads a Bing aerial image, runs SAM segmentation,
# and saves the detected objects as GeoJSON.
# Downloads a Bing aerial image, runs SAM segmentation,
# and saves the detected objects as GeoJSON.
from glidergun import stack

bing = stack("microsoft", (-123.164, 49.272, -123.162, 49.273), max_tiles=100)
bing.save("vancouver.tif")

sam = bing.sam("tree", "house", "car")
sam.to_geojson().save("vancouver.json")
```

Both examples access online services. Running SAM may require Hugging Face authorization or a model checkpoint, depending on your model setup.

## Library Overview

Glidergun is a Python library for geospatial raster analysis and map algebra. It brings NumPy-style expressions to georeferenced raster data, keeping coordinate reference system (CRS), extent, and cell size attached to each grid. It is designed for interactive exploration in notebooks as well as repeatable analysis scripts.

### Grids and stacks

The main objects are `Grid` and `Stack`. A `Grid` represents one raster band; a `Stack` groups aligned bands, such as the red, green, and blue channels of an image. Create them with `grid(...)` and `stack(...)` from NumPy arrays, raster files, Rasterio datasets, or supported remote sources.

### Map algebra and raster analysis

Use arithmetic and comparison operators directly on grids and stacks; bitwise operators are available for compatible data types. Operations can combine raster data with scalars or other grids. When grids share a CRS but differ in extent or cell size, Glidergun can align them for the operation. Conditional expressions, reclassification, range clipping, scaling, and band aggregation are also available.

For spatial analysis, Glidergun includes focal neighborhood statistics and filters, zonal statistics, interpolation, terrain derivatives such as slope and aspect, hillshade, and rasterization. Convert raster cells to point or polygon features, or burn geometries and points into a grid. Grids can also be clipped, resampled, reprojected, mosaicked, and processed in tiles.

### Data access and output

Read and write raster formats supported by Rasterio/GDAL. Search STAC catalogs such as Microsoft Planetary Computer for imagery and elevation data, or load supported web tile services and basemaps. Save results as raster files or serialize them to bytes; export vector features as GeoJSON.

### Imagery and interactive workflows

Stacks support per-band operations and conversion to NumPy arrays. Computer-vision workflows include image georeferencing with LoFTR and text-prompted segmentation with SAM. In notebooks, preview and animate grids and stacks; in supported ArcGIS Pro and QGIS sessions, add results to the active map.
