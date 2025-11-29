from glidergun import Grid, grid


def tick(grid: Grid):
    g = grid.focal_sum() - grid
    return (grid == 1) & (g == 2) | (g == 3)


def test_glidergun30():
    gosper = tick(grid("tests/input/glidergun30.asc"))
    md5s = set()
    while gosper.md5 not in md5s:
        md5s.add(gosper.md5)
        gosper = tick(gosper)
    assert len(md5s) == 60


def test_glidergun15():
    gosper = tick(grid("tests/input/glidergun15.asc"))
    md5s = set()
    while gosper.md5 not in md5s:
        md5s.add(gosper.md5)
        gosper = tick(gosper)
    assert len(md5s) == 30
