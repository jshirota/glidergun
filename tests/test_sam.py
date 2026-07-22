from glidergun import stack


def test_sam():
    bing = stack("microsoft", (-123.164, 49.272, -123.162, 49.273), max_tiles=10)
    sam = bing.sam("tree", "house", "car")
    geojson = sam.to_geojson()
    labels = {f.properties["label"] for f in geojson.features}
    assert labels == {"tree", "house", "car"}
