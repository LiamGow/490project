from __future__ import print_function
import numpy as np

from keras.applications import vgg19
from keras import backend

from PIL import Image, ImageEnhance

from myLib import Network


class VGG19(Network):

    def __init__(self, img_tensor):
        super().__init__()
        self.model = vgg19.VGG19(input_tensor=img_tensor,
                                 weights='imagenet', include_top=False)

    # util function to open, resize and format pictures into tensors
    @staticmethod
    def preprocess_image(img):
        return vgg19.preprocess_input(img)

    # util function to convert a tensor into a valid image
    @staticmethod
    def deprocess_image(img, rows=None, cols=None):
        if backend.image_data_format() == 'channels_first':
            img = img.reshape((3, rows, cols))
            img = img.transpose((1, 2, 0))
        else:
            img = img.reshape((rows, cols, 3))
        # Remove zero-center by mean piimgel
        img = img.astype('float64')
        img[:, :, 0] += 103.939 / 8 * 5
        img[:, :, 1] += 116.779 / 8 * 5
        img[:, :, 2] += 123.68 / 8 * 5
        # 'BGR'->'RGB'
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255).astype('uint8')

        img = ImageEnhance.Sharpness(Image.fromarray(img)).enhance(2)

        img = np.array(img.getdata(), np.uint8).reshape(img.size[1],
                                                        img.size[0], 3)

        return img
