# Map Algebra with NumPy

Inspired by the ARC/INFO GRID implementation of [Map Algebra](https://en.m.wikipedia.org/wiki/Map_algebra).

```
pip install glidergun
```

### Copernicus DEM

```python
from glidergun import grid

dem = grid("cop-dem-glo-90", (137.8, 34.5, 141.1, 36.8))
hillshade = dem.hillshade()

dem.save("dem.tif")
hillshade.save("hillshade.tif", "uint8")

```

<a href="https://github.com/jshirota/glidergun/blob/main/glidergun.ipynb" style="font-size:16px;">More Examples</a>

### License

This project is licensed under the MIT License.  See `LICENSE` for details.
