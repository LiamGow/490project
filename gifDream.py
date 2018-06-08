from keras import backend as K

from imgLib import *
from myLib import *

import sys

from png2gif import TimeLapse
from dreamLib import DeepDream
from inceptionV3Lib import InceptionV3


def simple_run(network, operation, output):
    check_args()
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


def check_args():
    if len(sys.argv) < 3:
        print("Usage: python3 png2gif.py <src_img> <#frames> [<rate>]")
        sys.exit()
    Cfg.img_nm = sys.argv[1]
    Cfg.img_nm_base = Cfg.img_nm[0:Cfg.img_nm.find('.')]
    Cfg.nframes = int(sys.argv[2])
    Cfg.rate = int(sys.argv[3]) if len(sys.argv) > 3 else 1


def main():
    simple_run(InceptionV3, DeepDream, TimeLapse)


if __name__ == "__main__":
    main()

