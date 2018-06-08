from myLib import *
from imgLib import *


class GifDream(Output):

    def __init__(self, args):
        self.load = load_gif
        self.args = args

    def run(self, gif, operation):
        gif_out_frames = []

        # deep dream algorithm
        print("Total frames", len(gif))
        for n, frame in enumerate(gif):
            print("Iteration {}/{}", n + 1, self.args.iterations)
            frm = operation.apply(frame, iterations=self.args.iterations)

            gif_out_frames.append(np.copy(frm))

        return gif_out_frames
