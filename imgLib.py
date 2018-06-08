from myLib import Cfg

import numpy as np
from keras.preprocessing import image
from PIL import Image
import imageio
import os


def save_img(path, img, postprocess_func=lambda i: i):
    pil_img = postprocess_func(np.copy(img))
    imageio.imwrite(path, pil_img)


def load_img(path, preprocess_func=lambda i: i, rows=None, cols=None):
    img = image.load_img(path) if not rows and cols\
        else image.load_img(path, rows, cols)
    img = preprocess_func(img)
    return img


def save_gif(path, frame_list, postprocess_func=lambda i: i):
    images = []
    filenames = []

    print("saving", path)

    for i, frame in enumerate(frame_list):
        filename = os.path.basename(path) + "_frame" + str(i) + ".png"
        filenames.append(filename)

        save_img(filename, frame, postprocess_func)

    for filename in filenames:
        images.append(imageio.imread(filename))

    imageio.mimsave(path, images, duration=0.1)

    for filename in filenames:
        os.remove(filename)


def load_gif(gif_path, preprocess_func=lambda i: i):
    imgs = []
    gif = Image.open(gif_path)
    try:
        while 1:
            # save gif frame as png
            fpath = os.path.basename(gif_path) + "_frame" + str(gif.tell()) + ".png"
            gif.save(fpath)

            # load png
            img = load_img(fpath, preprocess_func)
            os.remove(fpath)

            # append and loop
            imgs.append(np.copy(img))
            gif.seek(gif.tell() + 1)
    except EOFError:
        return imgs
