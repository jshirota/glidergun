from base64 import b64encode
from io import BytesIO
from typing import Optional, Union

import IPython
import matplotlib.pyplot as plt
import numpy as np

from glidergun._grid import Grid
from glidergun._stack import Stack
from glidergun._types import Extent


def _thumbnail(obj: Union[Grid, Stack], color, figsize=None):
    with BytesIO() as buffer:
        figure = plt.figure(figsize=figsize, frameon=False)
        axes = figure.add_axes((0, 0, 1, 1))
        axes.axis("off")

        n = 4000 / max(obj.width, obj.height)

        if n < 1:
            obj = obj.resample(obj.cell_size / n)

        obj = obj.to_uint8_range()

        if isinstance(obj, Grid):
            plt.imshow(obj.data, cmap=color)

        elif isinstance(obj, Stack):
            rgb = [
                obj.grids[i - 1].data for i in (color if color else (1, 2, 3))]
            alpha = np.where(np.isfinite(rgb[0] + rgb[1] + rgb[2]), 255, 0)
            plt.imshow(
                np.dstack([*[np.asanyarray(g, "uint8") for g in rgb], alpha]))

        plt.savefig(buffer, bbox_inches="tight", pad_inches=0)
        plt.close(figure)
        image = b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64, {image}"


def _map(
    obj: Union[Grid, Stack],
    color,
    opacity: float,
    basemap,
    width: int,
    height: int,
    attribution: Optional[str],
    grayscale: bool = True,
    **kwargs,
):
    import folium
    import jinja2

    obj_4326 = obj.project(4326)

    extent = Extent(
        obj_4326.xmin, max(
            obj_4326.ymin, -85), obj_4326.xmax, min(obj_4326.ymax, 85)
    )

    if obj_4326.extent != extent:
        obj_4326 = obj_4326.clip(*extent)

    obj_3857 = obj_4326.project(3857)

    figure = folium.Figure(width=str(width), height=str(height))
    bounds = [[obj_4326.ymin, obj_4326.xmin], [obj_4326.ymax, obj_4326.xmax]]

    if isinstance(basemap, str) or basemap is None:
        if basemap:
            tile_layer = folium.TileLayer(basemap, attr=attribution)
        else:
            tile_layer = folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="&copy; Esri",
            )

        options = {"zoom_control": False, **kwargs}
        folium_map = folium.Map(tiles=tile_layer, **options).add_to(figure)
        folium_map.fit_bounds(bounds)  # type: ignore

        if grayscale:
            macro = folium.MacroElement().add_to(folium_map)
            macro._template = jinja2.Template(
                f"""
                {{% macro script(this, kwargs) %}}
                tile_layer_{tile_layer._id}.getContainer()
                    .setAttribute("style", "filter: grayscale(100%); -webkit-filter: grayscale(100%);")
                {{% endmacro %}}
            """
            )
    else:
        folium_map = basemap

    folium.raster_layers.ImageOverlay(  # type: ignore
        image=_thumbnail(obj_3857, color, (20, 20)),
        bounds=bounds,
        opacity=opacity,
    ).add_to(folium_map)

    return folium_map


ipython = IPython.get_ipython()  # type: ignore

if ipython:

    def html(obj: Union[Grid, Stack]):
        description = str(obj).replace("|", "<br />")
        if isinstance(obj, Grid):
            thumbnail = _thumbnail(obj, obj._cmap)
            extent = obj.extent
        elif isinstance(obj, Stack):
            thumbnail = _thumbnail(obj, obj._rgb)
            extent = obj.extent
        return f'<div>{description}</div><img src="{thumbnail}" /><div>{extent}</div>'

    formatters = ipython.display_formatter.formatters  # type: ignore
    formatter = formatters["text/html"]
    formatter.for_type(Grid, html)
    formatter.for_type(Stack, html)
    formatter.for_type(
        tuple,
        lambda items: (
            f"""
            <table>
                <tr style="text-align: left">
                    {"".join(f"<td>{html(item)}</td>" for item in items)}
                </tr>
            </table>
        """
            if all(isinstance(item, Grid) or isinstance(item, Stack) for item in items)
            else f"{items}"
        ),
    )
