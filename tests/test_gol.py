from glidergun import Grid, grid


def tick(grid: Grid):
    g = grid.focal_sum() - grid
    return (grid == 1) & (g == 2) | (g == 3)


def test_glidergun():
    gosper = tick(grid("tests/input/glidergun.asc"))
    md5s = set()
    while gosper.sha256 not in md5s:
        md5s.add(gosper.sha256)
        gosper = tick(gosper)
    assert len(md5s) == 60
