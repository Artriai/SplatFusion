from splatfusion.preprocess import map_original_pixels_to_vggt


def test_pixel_mapping_identity_when_no_crop_resize_or_padding():
    meta = {
        "crop_left": 0, "crop_top": 0, "crop_width": 100, "crop_height": 80,
        "resized_width": 100, "resized_height": 80, "pad_left": 0, "pad_top": 0,
    }
    u, v = map_original_pixels_to_vggt([0, 20, 99], [0, 30, 79], meta)
    assert list(u) == [0, 20, 99]
    assert list(v) == [0, 30, 79]


def test_pixel_mapping_accounts_for_crop_and_padding():
    meta = {
        "crop_left": 10, "crop_top": 5, "crop_width": 80, "crop_height": 60,
        "resized_width": 160, "resized_height": 120, "pad_left": 8, "pad_top": 4,
    }
    u, v = map_original_pixels_to_vggt([10], [5], meta)
    assert abs(u[0] - 8.5) < 1e-8
    assert abs(v[0] - 4.5) < 1e-8
