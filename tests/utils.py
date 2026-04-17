from glidergun.types import BBox


def extents_equal(extent1: BBox, extent2: BBox, tolerance: float = 1e-6):
    return all(abs(a - b) < tolerance for a, b in zip(extent1, extent2, strict=True))
