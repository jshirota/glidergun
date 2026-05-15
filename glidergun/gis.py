import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING

from glidergun.utils import safe_filename

if TYPE_CHECKING:
    from glidergun.geojson import FeatureCollection
    from glidergun.grid import Grid
    from glidergun.stack import Stack


def add_to_map(data: "Grid | Stack | FeatureCollection", name: str | None):
    from glidergun.geojson import FeatureCollection

    file = name or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def save(folder: str):
        if isinstance(data, FeatureCollection):
            file_path = os.path.join(folder, file + ".json")
        else:
            file_path = os.path.join(folder, file + ".tif")
        file_path = safe_filename(file_path)
        data.save(file_path)
        return file_path

    if "arcpy" in sys.modules:
        try:
            import arcpy  # type: ignore

            aprx = arcpy.mp.ArcGISProject("CURRENT")
            folder = arcpy.env.scratchFolder
            file_path = save(folder)

            if isinstance(data, FeatureCollection):
                arcpy.conversion.JSONToFeatures(
                    in_json_file=file_path, out_features=os.path.join(folder, file), geometry_type="POLYGON"
                )
            else:
                aprx.activeMap.addDataFromPath(file_path)
        except Exception as ex:
            print(f"Failed to add layer to active map: {ex}")
        return

    if "qgis.core" in sys.modules:
        try:
            from qgis.core import QgsProcessingUtils, QgsProject, QgsRasterLayer, QgsVectorLayer  # type: ignore

            folder = QgsProcessingUtils.tempFolder()
            file_path = save(folder)

            if isinstance(data, FeatureCollection):
                QgsProject.instance().addMapLayer(QgsVectorLayer(file_path, file + ".json", "ogr"))
            else:
                QgsProject.instance().addMapLayer(QgsRasterLayer(file_path, file + ".tif"))
        except Exception as ex:
            print(f"Failed to add layer to active map: {ex}")
        return

    raise RuntimeError("This function is only supported within ArcGIS Pro or QGIS.")
