from imgLib import *
from keras import backend as K


class Cfg:
    src_dir = "./imsrc/"
    out_dir = "./imout/"
    abstraction_lvl = 1
    max_loss = None
    grad_step = 0.01
    iterct = 25

    img_nm = None
    img_nm_base = None
    nframes = None
    rate = 1


class Network:

    def __init__(self):
        self.model = None

    @staticmethod
    def preprocess(img):
        pass

    @staticmethod
    def postprocess(img):
        pass


class Operation:

    def apply(self, img, iterations):
        pass


class Output:

    def __init__(self):
        self.load = None

    def run(self, img, operation):
        pass


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