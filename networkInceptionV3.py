import numpy as np

from keras.applications import inception_v3
from keras import backend
from keras.preprocessing import image

from myLib import Network


class InceptionV3(Network):

    def __init__(self):
        super().__init__()
        self.model = inception_v3.InceptionV3(weights='imagenet',
                                              include_top=False)

    @staticmethod
    def preprocess_image(img):
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return inception_v3.preprocess_input(img)

    @staticmethod
    def deprocess_image(img):
        if backend.image_data_format() == 'channels_first':
            img = img.reshape((3, img.shape[2], img.shape[3]))
            img = img.transpose((1, 2, 0))
        else:
            img = img.reshape((img.shape[1], img.shape[2], 3))
        img /= 2.
        img += 0.5
        img *= 255.
        img = np.clip(img, 0, 255).astype('uint8')
        return img
