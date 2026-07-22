from glidergun import stack


def test_loftr():
    bing = stack("microsoft", (-123.28, 49.21, -123.07, 49.34), max_tiles=100)
    vancouver = stack("data/vancouver.png")
    homography = vancouver.georeference_to(bing, homography_only=True)
    assert homography is not None
