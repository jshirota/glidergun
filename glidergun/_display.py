from collections.abc import Iterable
from typing import Any

import IPython
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from glidergun._grid import Grid
from glidergun._literals import ColorMap
from glidergun._sam import SamResult
from glidergun._stack import Stack


def get_html(obj: Grid | Stack | ArtistAnimation | SamResult):
    if isinstance(obj, ArtistAnimation):
        return f"<div>{obj.to_jshtml()}</div>"
    if isinstance(obj, SamResult):
        return get_html(obj.highlight())
    n = 100
    description = "<br />".join(s if len(s) <= n else s[:n] + "..." for s in str(obj).split("|"))
    return f'<div><div>{description}</div><img src="{obj.img}" /></div>'


def animate(
    grids: Iterable[Grid | Stack],
    cmap: ColorMap | Any = "gray",
    interval: int = 200,
):
    first = next(iter(grids))
    n = 5 / first.extent.width
    figsize = (first.extent.width * n, first.extent.height * n)

    def iterate():
        yield first
        yield from grids

    figure = plt.figure(figsize=figsize, frameon=False)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.axis("off")
    frames = [
        [axes.imshow(o.to_array() if isinstance(o, Stack) else o.data, cmap=cmap, animated=True, aspect="auto")]
        for o in iterate()
    ]
    plt.close()
    return ArtistAnimation(figure, frames, interval=interval, blit=True)


if ipython := IPython.get_ipython():  # type: ignore
    formatters = ipython.display_formatter.formatters  # type: ignore
    formatter = formatters["text/html"]
    formatter.for_type(Grid, get_html)
    formatter.for_type(Stack, get_html)
    formatter.for_type(ArtistAnimation, get_html)
    formatter.for_type(SamResult, get_html)
    formatter.for_type(
        tuple,
        lambda items: (
            f"""
            <table>
                <tr style="text-align: left">
                    {"".join(f"<td>{get_html(item)}</td>" for item in items)}
                </tr>
            </table>
            """
            if all(isinstance(item, (Grid | Stack | SamResult)) for item in items)
            else f"{items}"
        ),
    )
