# Map Algebra with NumPy

Inspired by the ARC/INFO GRID implementation of [Map Algebra](https://en.m.wikipedia.org/wiki/Map_algebra).

```
pip install glidergun
```

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

animate(simulate(seed), interval=40)
```

<img src="game_of_life.gif" width="600"/>

<a href="glidergun.ipynb" style="font-size:16px;">More Examples</a>

### License

This project is licensed under the MIT License.  See `LICENSE` for details.
