import math


def coords_to_quadkey(lon: float, lat: float, zoom: int):
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n)
    quadkey = ""
    for _ in range(zoom, 0, -1):
        bit = (x & 1) + 2 * (y & 1)
        quadkey = str(bit) + quadkey
        x >>= 1
        y >>= 1
    return quadkey


def quadkey_to_tile(quadkey: str):
    x = 0
    y = 0
    zoom = len(quadkey)
    for i, digit in enumerate(reversed(quadkey)):
        bit = int(digit)
        mask = 1 << i
        if bit & 1:
            x |= mask
        if bit & 2:
            y |= mask
    return x, y, zoom


def tile_to_bbox(x: int, y: int, zoom: int):
    n = 2**zoom
    lon_min: float = x / n * 360.0 - 180.0
    lon_max: float = (x + 1) / n * 360.0 - 180.0
    lat_min: float = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat_max: float = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon_min, lat_min, lon_max, lat_max


def tile_to_quadkey(x: int, y: int, zoom: int):
    quadkey = ""
    for i in range(zoom, 0, -1):
        bit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            bit += 1
        if (y & mask) != 0:
            bit += 2
        quadkey += str(bit)
    return quadkey


def get_tile_count_at_zoom(extent: tuple[float, float, float, float] | list[float], zoom: int):
    xmin, ymin, xmax, ymax = extent
    x1, y1, _ = quadkey_to_tile(coords_to_quadkey(xmin, ymin, zoom))
    x2, y2, _ = quadkey_to_tile(coords_to_quadkey(xmax, ymax, zoom))
    return (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)


def get_tiles_at_zoom(extent: tuple[float, float, float, float] | list[float], zoom: int):
    xmin, ymin, xmax, ymax = extent
    x1, y1, _ = quadkey_to_tile(coords_to_quadkey(xmin, ymin, zoom))
    x2, y2, _ = quadkey_to_tile(coords_to_quadkey(xmax, ymax, zoom))
    tiles = [
        (x, y, tile_to_quadkey(x, y, zoom))
        for x in range(min(x1, x2), max(x1, x2) + 1)
        for y in range(min(y1, y2), max(y1, y2) + 1)
    ]
    return tiles


def get_tiles(extent: tuple[float, float, float, float] | list[float], max_tiles: int, max_zoom: int):
    zoom = max_zoom
    while get_tile_count_at_zoom(extent, zoom) > max_tiles:
        zoom -= 1
    tiles = get_tiles_at_zoom(extent, zoom)
    return [(x, y, zoom, q, *tile_to_bbox(x, y, zoom)) for x, y, q in tiles]
