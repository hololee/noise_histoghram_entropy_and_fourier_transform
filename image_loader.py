from imageio import imread
import os


class ImageLoader(object):
    def __init__(self, file_name):
        self.img = imread(os.path.join('.', file_name))

    def __call__(self):
        print("Image load : {}".format(self.img.shape))
        return self.img
