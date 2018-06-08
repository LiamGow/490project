from myLib import *
from imgLib import *


class GifDream(Output):

    def __init__(self, args):
        self.load = load_gif
        self.args = args

    def run(self, gif, operation):
        gif_out_frames = []

        for n, frame in enumerate(gif):

            print("Frame {}/{}".format(n + 1, self.args.iterations))
            img = operation.apply(frame, iterations=self.args.iterations)

            # save frame
            filename = os.path.splitext(os.path.basename(
                self.args.image_path))[0] + "_frame" + str(n) + ".png"
            save_img(filename, img, self.args.network.deprocess_image)

            gif_out_frames.append(filename)

        return gif_out_frames
