from myLib import Cfg

import numpy as np

from keras.applications import inception_v3
from keras import backend as K
from keras.preprocessing import image
from PIL import Image
import imageio

def save_gif(frame_list, fname):
    print("saving", fname)
    images = []
    filenames = []
    for i, frame in enumerate(frame_list):
        filename = "gif_frames/" + "frame" + str(i) + ".png"
        save_img(frame, filename)
        filenames.append(filename)
    for filename in filenames:
        images.append(imageio.imread(Cfg.out_dir + filename))
    imageio.mimsave(Cfg.out_dir + fname, images, duration=0.1)

def preprocess_gif(gif_path):
    imgs = []
    gif = Image.open(gif_path)
    try:
        while 1:
            fpath = Cfg.out_dir + "gif_frames/frame" + str(gif.tell()) + ".png"
            gif.save(fpath)
            img = image.load_img(fpath)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = inception_v3.preprocess_input(img)
            imgs.append(np.copy(img))
            gif.seek(gif.tell() + 1)
    except EOFError:
        return imgs


def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_img(img, fname, deprocess_func=deprocess_image):
    pil_img = deprocess_func(np.copy(img))
    imageio.imwrite(Cfg.out_dir + fname, pil_img)