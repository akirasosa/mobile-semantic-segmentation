import numpy as np
from albumentations import CoarseDropout


def cutout(img, holes):
    # Make a copy of the input image since we don't want to modify it directly
    img = img.copy()
    for x1, y1, x2, y2 in holes:
        img[y1:y2, x1:x2] = np.random.randint(0, 255, size=(y2 - y1, x2 - x1, 3))
    return img


class MyCoarseDropout(CoarseDropout):
    def apply(self, image, fill_value=0, holes=None, **params):
        if holes is None:
            holes = []
        return cutout(image, holes)
