# Map Algebra with NumPy

Inspired by the ARC/INFO GRID implementation of [Map Algebra](https://en.m.wikipedia.org/wiki/Map_algebra).

### Basic Usage

```bash
pip install glidergun
```

```python
from glidergun import grid

dem = grid("cop-dem-glo-90", (137.8, 34.5, 141.1, 36.8))
hillshade = dem.hillshade()

dem.save("dem.tif")
hillshade.save("hillshade.tif", "uint8")
```

### With Segment Anything Model (larger dependency download)

**CPU-only:**
```bash
pip install glidergun[torch]
```

**GPU (NVIDIA CUDA 12.1):**

With pip:
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 glidergun[torch]
```

With uv:
```bash
uv pip install glidergun[torch]
```

Note: GPU acceleration requires Python 3.12 or 3.13. For other Python versions or different CUDA versions, see [PyTorch install guide](https://pytorch.org/get-started/locally/).

```python
from glidergun import stack

bing = stack("microsoft", (-123.164, 49.272, -123.162, 49.273), max_tiles=100)
bing.save("vancouver.tif")

sam = bing.sam("tree", "house", "car")
sam.to_geojson().save("vancouver.json")
```

### License

This project is licensed under the MIT License.  See `LICENSE` for details.
