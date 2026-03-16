from collections.abc import Iterable
from functools import wraps
from io import BytesIO

from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from numpy import ndarray

from glidergun.types import Chart


def with_agg_backend(func):
    import matplotlib
    import matplotlib.pyplot as plt

    @wraps(func)
    def wrapped(*args, **kwargs):
        old_backend = plt.get_backend()
        matplotlib.use("Agg")
        try:
            return func(*args, **kwargs)
        finally:
            matplotlib.use(old_backend)

    return wrapped


@with_agg_backend
def create_thumbnail(data, cmap=None, figsize: tuple[float, float] | None = (5, 5)) -> bytes:
    with BytesIO() as buffer:
        figure = plt.figure(figsize=figsize, frameon=False)
        axes = figure.add_axes((0, 0, 1, 1))
        axes.axis("off")
        plt.imshow(data, cmap=cmap)
        plt.savefig(buffer, bbox_inches="tight", pad_inches=0)
        plt.close(figure)
        return buffer.getvalue()


@with_agg_backend
def create_histogram(*data: tuple[dict[float, int], str]) -> Chart:
    """Build a histogram chart of value counts (NaN-aware)."""
    figure, axes = plt.subplots()
    for bins, color in reversed(data):
        axes.bar(list(bins.keys()), list(bins.values()), color=color)
    return Chart(figure, axes)


@with_agg_backend
def create_animation(frames: Iterable[ndarray], cmap=None, interval: int = 200) -> ArtistAnimation:
    first = next(iter(frames))
    n = 5 / first.shape[1]
    figsize = (first.shape[1] * n, first.shape[0] * n)
    figure = plt.figure(figsize=figsize, frameon=False)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.axis("off")
    artists = [[axes.imshow(array, cmap=cmap, animated=True, aspect="auto")] for array in frames]
    plt.close(figure)
    return ArtistAnimation(figure, artists, interval=interval, blit=True)
