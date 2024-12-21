# Map Algebra with NumPy

```
pip install glidergun
```

### Creating a hillshade from SRTM DEM

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

![](image1.png)

### Calculating the NDVI from Landsat bands

```python
from glidergun import grid

band4 = grid(".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B4.TIF")
band5 = grid(".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B5.TIF")

ndvi = (band5 - band4) / (band5 + band4)

ndvi.plot("gist_earth")
```

![](image2.png)

### Conway's Game of Life

```python
from glidergun import animate, grid


def tick(g):
    count = g.focal_sum() - g
    return (g == 1) & (count == 2) | (count == 3)


def simulate(g):
    md5s = set()
    while g.md5 not in md5s:
        md5s.add(g.md5)
        yield -(g := tick(g))


seed = grid((120, 80)).randomize() < 0.5

animation = animate(simulate(seed), interval=40)

# animation.save("game_of_life.gif")

animation
```

<img src="game_of_life.gif" width="600"/>
