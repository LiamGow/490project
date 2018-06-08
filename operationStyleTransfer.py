from __future__ import print_function

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg19
from keras import backend as K, models

from PIL import Image, ImageEnhance

import imgLib
from myLib import Operation


class StyleTransfer(Operation):

    def __init__(self, args, model):
        pass