import numpy as np
from splatfusion.masking import _select_automatic_mask


def test_select_automatic_mask_prefers_centered_object():
    h = w = 100
    centered = np.zeros((h, w), dtype=bool)
    centered[25:75, 25:75] = True
    corner = np.zeros((h, w), dtype=bool)
    corner[0:50, 0:50] = True
    annotations = [
        {"segmentation": corner, "predicted_iou": 0.95, "stability_score": 0.95},
        {"segmentation": centered, "predicted_iou": 0.9, "stability_score": 0.9},
    ]
    selected = _select_automatic_mask(annotations, h, w)
    assert np.array_equal(selected, centered)
