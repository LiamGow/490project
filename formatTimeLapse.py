from myLib import *
from imgLib import *


class TimeLapse(Output):

    def __init__(self, args):
        self.load = load_img
        self.args = args

    def run(self, frame, operation):
        gif_out_frames = []

        try:
            for n in range(self.args.iterations):

                print("Frame {}/{}".format(n + 1, self.args.iterations))
                frame = operation.apply(frame, iterations=self.args.rate)

                # save frame
                filename = self.args.out_prefix + os.path.splitext(os.path.basename(
                    self.args.image_path))[0] + "_frame" + str(n) + ".png"
                save_img(filename, frame, self.args.network.deprocess_image)

                gif_out_frames.append(filename)

        except(ResourceExhaustedError):
            return gif_out_frames

        return gif_out_frames