from myLib import Cfg

import numpy as np
from keras.preprocessing import image
from PIL import Image
import imageio


def save_img(path, img, postprocess_func=lambda i: i):
    pil_img = postprocess_func(np.copy(img))
    imageio.imwrite(path, pil_img)


def load_img(path, preprocess_func=lambda i: i, rows=None, cols=None):
    img = image.load_img(path) if not rows and cols\
        else image.load_img(path, rows, cols)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_func(img)
    return img


def save_gif(fname, frame_list, postprocess_func=lambda i: i):
    images = []
    filenames = []

    print("saving", fname)

    for i, frame in enumerate(frame_list):
        filename = "gif_frames/" + "frame" + str(i) + ".png"
        filenames.append(filename)

        save_img(filename, frame, postprocess_func)

    for filename in filenames:
        images.append(imageio.imread(Cfg.out_dir + filename))

    imageio.mimsave(Cfg.out_dir + fname, images, duration=0.1)


def load_gif(gif_path, preprocess_func=lambda i: i):
    imgs = []
    gif = Image.open(gif_path)
    try:
        while 1:
            # save gif frame as png
            fpath = Cfg.out_dir + "gif_frames/frame" + str(gif.tell()) + ".png"
            gif.save(fpath)

            # load png
            img = load_img(fpath, preprocess_func)

            # append and loop
            imgs.append(np.copy(img))
            gif.seek(gif.tell() + 1)
    except EOFError:
        return imgs
