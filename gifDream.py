from myLib import *
from imgLib import *

import sys


class GifDream(Output):

    def __init__(self):
        self.load = load_gif

    def run(self, gif, operation):
        gif_out_frames = []

        # deep dream algorithm
        print("Total frames", len(gif))
        for i, frame in enumerate(gif):
            print("Processing frame ", i)
            frm = operation.apply(frame, iterations=Cfg.iterct)

            gif_out_frames.append(np.copy(frm))

        return gif_out_frames


def check_args():
    if len(sys.argv) != 2:
        print("Usage: python3 gif2gif.py <src_gif>")
        sys.exit()
    Cfg.img_nm = sys.argv[1]
    Cfg.img_nm_base = Cfg.img_nm[0:Cfg.img_nm.find('.')]
