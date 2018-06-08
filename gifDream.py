from keras import backend as K

from imgLib import *
from myLib import *


def simple_run(network, operation, output):
    K.set_learning_phase(0)  # disables all training specific operations

    net = network()
    op = operation(net.model)
    out = output()

    # get the input img
    img = out.load(Cfg.src_dir + Cfg.img_nm, net.preprocess_image)

    # deep dream algorithm
    gif = out.run(img, op)

    # write out the dream sequence as gif
    save_gif(Cfg.img_nm_base + "_dreaming.gif", gif, net.deprocess_image)
