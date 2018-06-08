from myLib import *
from imgLib import *


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
