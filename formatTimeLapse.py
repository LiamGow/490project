from myLib import *
from imgLib import *


class TimeLapse(Output):

    def __init__(self, args):
        self.load = load_img
        self.args = args

    def run(self, img, operation):
        gif_out_frames = []

        for n in range(self.args.iterations):

            # run frame operation
            print("Iteration {}/{}", n + 1, self.args.iterations)
            img = operation.apply(img, iterations=self.args.rate)

            # save frame
            filename = os.path.splitext(os.path.basename(
                self.args.image_path))[0] + "_frame" + str(n) + ".png"
            save_img(filename, img, self.args.network.deprocess_image)

            gif_out_frames.append(filename)

        return gif_out_frames
