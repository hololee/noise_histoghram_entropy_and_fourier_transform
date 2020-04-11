from imageio import imread
import os
from PIL import Image
import numpy as np


class ImageLoader(object):
    def __init__(self, file_name):
        self.img = imread(os.path.join('.', file_name))

    def __call__(self, resize_factor=None):
        print("Image load : {}".format(self.img.shape))

        if resize_factor is None:
            return self.img
        else:
            temp_image = Image.fromarray(self.img)
            temp_image = temp_image.resize(
                (int(self.img.shape[0] * resize_factor), int(self.img.shape[1] * resize_factor)))
            return np.array(temp_image)
