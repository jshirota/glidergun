# Map Algebra with NumPy

```
pip install glidergun
```

```python
from glidergun import grid, mosaic

dem1 = grid(".data/n55_e008_1arc_v3.bil")
dem2 = grid(".data/n55_e009_1arc_v3.bil")

dem = mosaic(dem1, dem2)
hillshade = dem.hillshade()

# hillshade.save(".output/hillshade.tif", "uint8")
# hillshade.save(".output/hillshade.png")
# hillshade.save(".output/hillshade.kmz")

dem, hillshade
```

![alt text](image.png)
