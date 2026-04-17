from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Union

import IPython

from glidergun.grid import Grid
from glidergun.literals import ColorMap
from glidergun.plot import create_animation
from glidergun.sam import SamResult
from glidergun.stack import Stack

if TYPE_CHECKING:
    from matplotlib.animation import ArtistAnimation


def get_html(obj: Union[Grid, Stack, "ArtistAnimation", SamResult]):
    if isinstance(obj, ArtistAnimation):
        return f"<div>{obj.to_jshtml()}</div>"
    if isinstance(obj, SamResult):
        return get_html(obj.highlight())
    n = 100
    description = "<br />".join(s if len(s) <= n else s[:n] + "..." for s in str(obj).split("|"))
    return f'<div><div>{description}</div><img src="{obj.img}" /></div>'


def animate(grids: Iterable[Grid | Stack], cmap: ColorMap | Any = "gray", interval: int = 200):
    return create_animation([(o.to_array() if isinstance(o, Stack) else o.data) for o in grids], cmap, interval)


if ipython := IPython.get_ipython():  # type: ignore
    from matplotlib.animation import ArtistAnimation

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
