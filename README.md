# Map Algebra with NumPy

Inspired by the ARC/INFO GRID implementation of [Map Algebra](https://en.m.wikipedia.org/wiki/Map_algebra).

## Basic Usage

```bash
pip install glidergun
```

### Example:

```python
from glidergun import grid

dem = grid("cop-dem-glo-90", (137.8, 34.5, 141.1, 36.8))
hillshade = dem.hillshade()

dem.save("dem.tif")
hillshade.save("hillshade.tif", "uint8")
```

## With Segment Anything Model (larger dependency download)

### CPU-only:

```bash
pip install glidergun[torch]
```

### GPU (NVIDIA CUDA):

Python 3.14:
```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
pip install glidergun[torch]
```

Python 3.10 ~ 3.13:
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install glidergun[torch]
```

### Example:

```python
from glidergun import stack

bing = stack("microsoft", (-123.164, 49.272, -123.162, 49.273), max_tiles=100)
bing.save("vancouver.tif")

sam = bing.sam("tree", "house", "car")
sam.to_geojson().save("vancouver.json")
```

## ArcGIS

```
"%LOCALAPPDATA%\ESRI\conda\envs\arcgispro-py3-clone\python.exe" -m pip uninstall -y glidergun torch torchvision
"%LOCALAPPDATA%\ESRI\conda\envs\arcgispro-py3-clone\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
"%LOCALAPPDATA%\ESRI\conda\envs\arcgispro-py3-clone\python.exe" -m pip install glidergun[torch]

```

## QGIS

```
"C:\Program Files\QGIS 3.44.10\apps\Python312\python.exe" -m pip uninstall -y glidergun torch torchvision
"C:\Program Files\QGIS 3.44.10\apps\Python312\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
"C:\Program Files\QGIS 3.44.10\apps\Python312\python.exe" -m pip install glidergun[torch]
```
