from myLib import *
from imgLib import *


class GifDream(Output):

    def __init__(self, args):
        self.load = load_gif
        self.args = args

    def run(self, gif, operation):
        gif_out_frames = []

        try:
            for n, frame in enumerate(gif):

                print("Frame {}/{}".format(n + 1, len(gif)))
                frame = operation.apply(frame, iterations=self.args.iterations)

                # save frame
                filename = self.args.out_prefix + os.path.splitext(os.path.basename(
                    self.args.image_path))[0] + "_frame" + str(n) + ".png"
                save_img(filename, frame, self.args.network.deprocess_image)

                del frame

                gif_out_frames.append(filename)

        except(ResourceExhaustedError):
            return gif_out_frames

        return gif_out_frames
