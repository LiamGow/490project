from myLib import *
from imgLib import *


class TimeLapse(Output):

    def __init__(self, args):
        self.load = load_img
        self.args = args

    def run(self, img, operation):
        gif_out_frames = []
        for n in range(self.args.iterations):
            img = operation.apply(img, iterations=self.args.rate)
            gif_out_frames.append(np.copy(img))

        return gif_out_frames
