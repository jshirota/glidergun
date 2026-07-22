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

```bash
# PyPI
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install glidergun
```

```bash
# UV
uv add torch torchvision --index pytorch-cu130=https://download.pytorch.org/whl/cu130
uv add glidergun
```

## Examples

```python
# Downloads a Copernicus DEM and creates a hillshade GeoTIFF.
from glidergun import grid

dem = grid("cop-dem-glo-90", (137.8, 34.5, 141.1, 36.8))
hillshade = dem.hillshade()

dem.save("dem.tif")
hillshade.save("hillshade.tif", "uint8")
```

```python
# Downloads a Bing aerial image, runs SAM segmentation,
# and saves the detected objects as GeoJSON.
from glidergun import stack

bing = stack("microsoft", (-123.164, 49.272, -123.162, 49.273), max_tiles=100)
bing.save("vancouver.tif")

sam = bing.sam("tree", "house", "car")
sam.to_geojson().save("vancouver.json")
```
