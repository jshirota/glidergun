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

hillshade
```

![DEM](https://raw.githubusercontent.com/jshirota/glidergun/main/dem.png)

### With Segment Anything Model (larger dependency download)

```bash
pip install glidergun[sam]
```

```python
from glidergun import stack

url = "https://t.ssl.ak.tiles.virtualearth.net/tiles/a{q}.jpeg?g=15437"
bing = stack(url, (-123.164, 49.272, -123.162, 49.273), max_tiles=100)
bing.save("vancouver.tif")

sam = bing.sam("tree", "house", "car")
sam.to_geojson().save("vancouver.json")
```

![QGIS](https://raw.githubusercontent.com/jshirota/glidergun/main/qgis.png)

<a href="https://github.com/jshirota/glidergun/blob/main/glidergun.ipynb" style="font-size:16px;">More Examples</a>

### License

This project is licensed under the MIT License.  See `LICENSE` for details.
