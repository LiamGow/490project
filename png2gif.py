from myLib import *
from imgLib import *

from dreamLib import DeepDream
from inceptionV3Lib import InceptionV3

from keras import backend as K

import sys

def main():
    check_args()
    K.set_learning_phase(0)  # disables all training specific operations

    network = InceptionV3()
    operation = DeepDream(network.model)

    # get the input img
    img_path = Cfg.src_dir + Cfg.img_nm
    img = load_img(img_path, network.preprocess_image)

    # deep dream algorithm
    gif_out_frames = []
    while (Cfg.nframes > 0):
        img = operation.apply(img, iterations=Cfg.rate)
        gif_out_frames.append(np.copy(img))
        Cfg.nframes -= 1

    # write out the dream sequence as gif
    save_gif(Cfg.img_nm_base + "_dreaming.gif", gif_out_frames, network.deprocess_image)


def check_args():
    if len(sys.argv) < 4:
        print("Usage: python3 png2gif.py <src_img> <#frames> [<rate>]")
        sys.exit()
    Cfg.img_nm = sys.argv[1]
    Cfg.img_nm_base = Cfg.img_nm[0:Cfg.img_nm.find('.')]
    Cfg.nframes = int(sys.argv[2])
    Cfg.rate = int(sys.argv[3]) if len(sys.argv) > 4 else 1

if __name__ == "__main__":
    main()
