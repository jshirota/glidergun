import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import IO, Any

from shapely.geometry import Point, Polygon, mapping

from glidergun._utils import create_directory_for

type Geometry = Point | Polygon


@dataclass(frozen=True)
class Feature[T: Geometry]:
    geometry: T
    properties: dict[str, Any]


class FeatureCollection[T: Geometry]:
    def __init__(self, features: Iterable[tuple[T, dict]]):
        self.features = [Feature(g, p) for g, p in features]

    def bind[TOutput: Geometry](self, func: Callable[[Feature[T]], Iterable[Feature[TOutput]]]):
        features: list[Feature[TOutput]] = []
        for feature in self.features:
            features.extend(func(feature))
        return FeatureCollection((f.geometry, f.properties) for f in features)

    def filter(self, func: Callable[[Feature[T]], bool]):
        return self.bind(lambda f: [f] if func(f) else [])

    def update_geometry[TOutput: Geometry](self, func: Callable[[T], TOutput]):
        return self.bind(lambda f: [Feature(func(f.geometry), f.properties)])

    def save(self, file: str | IO[str]):
        d = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(feature.geometry),
                    "properties": feature.properties,
                }
                for feature in self.features
            ],
        }
        if isinstance(file, str):
            create_directory_for(file)
            with open(file, "w") as f:
                json.dump(d, f)
        else:
            json.dump(d, file)
