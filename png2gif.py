from myLib import *
from imgLib import *

from dreamLib import DeepDream
from inceptionV3Lib import InceptionV3

from gifDream import simple_run

from keras import backend as K

import sys


class TimeLapse(Output):

    def __init__(self):
        self.load = load_img

    def run(self, img, operation):
        gif_out_frames = []
        while Cfg.nframes > 0:
            img = operation.apply(img, iterations=Cfg.rate)
            gif_out_frames.append(np.copy(img))
            Cfg.nframes -= 1

        return gif_out_frames


def main():
    check_args()
    simple_run(InceptionV3, DeepDream, TimeLapse)


def check_args():
    if len(sys.argv) < 3:
        print("Usage: python3 png2gif.py <src_img> <#frames> [<rate>]")
        sys.exit()
    Cfg.img_nm = sys.argv[1]
    Cfg.img_nm_base = Cfg.img_nm[0:Cfg.img_nm.find('.')]
    Cfg.nframes = int(sys.argv[2])
    Cfg.rate = int(sys.argv[3]) if len(sys.argv) > 3 else 1

if __name__ == "__main__":
    main()
