def extents_equal(
    extent1: tuple[float, float, float, float],
    extent2: tuple[float, float, float, float],
    tolerance: float = 1e-6,
):
    return all(abs(a - b) < tolerance for a, b in zip(extent1, extent2, strict=True))
