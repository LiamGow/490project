import argparse

from keras import backend

from imgLib import *

# networks
from myLib import Network
from networkInceptionV3 import InceptionV3
from networkVGG19 import VGG19

# operations
from operationDeepDream import DeepDream
from operationGoogleDream import GoogleDream # only runs with Network
from operationStyleTransfer import StyleTransfer

# formats
from formatGifDream import GifDream
from formatTimeLapse import TimeLapse


def simple_run(network, operation, format, args):
    backend.set_learning_phase(0)  # disables all training specific operations

    net = network()
    op = operation(args, net.model)
    form = format(args)

    # get the input img
    img = form.load(args.image_path, net.preprocess_image)

    # deep dream algorithm
    gif = form.run(img, op)

    # write out the dream sequence as gif
    base = os.path.splitext(os.path.basename(args.image_path))[0]
    save_gif(os.path.join(args.out_prefix, base) + ".gif", gif, net.deprocess_image)

def get_args():
    parser = argparse.ArgumentParser(description='Keras gif dreamer.')

    # config arguments
    parser.add_argument('network', type=str,
                        help='Network to use.')
    parser.add_argument('operation', type=str,
                        help='Operation to perform on each frame.')
    parser.add_argument('format', type=str,
                        help='Input/Output format.')

    # i/o arguments
    parser.add_argument('image_path', type=str,
                        help='Path to the image/gif to transform.')
    parser.add_argument('-o', '--out_prefix', type=str, required=False,
                        help='Prefix for the saved results.')

    # general arguments
    parser.add_argument('-i', '--iterations', type=int, default=10, required=False,
                        help='Number of iterations to run.')
    parser.add_argument('-t', '--step', type=float, default=0.01, required=False,
                        help='Tweak step to gradient ascend by.')

    # deep dream arguments
    parser.add_argument('-a', '--abstraction', type=float, default=1, required=False,
                        help='Higher numbers maximize larger features in loss.')
    parser.add_argument('-m', '--max_loss', type=float, default=0, required=False,
                        help='Tweak step to gradient ascend by.')

    # style transfer arguments
    parser.add_argument('-s', '--style', type=str,
                        help='Path to the style reference image.')
    parser.add_argument('-c', '--content_weight', type=float, default=0.025,
                        required=False,
                        help='Content weight.')
    parser.add_argument('-w', '--style_weight', type=float, default=1.0,
                        required=False,
                        help='Style weight.')
    parser.add_argument('-v', '--variation', type=float, default=1.0, required=False,
                        help='Total Variation weight.')
    parser.add_argument('-w', '--width', type=int, default=None, required=False,
                        help='Output image width.')

    # google dream arguments
    parser.add_argument('--octaves', type=int, default=4, required=False,
                        help='Octaves per iteration.')

    # time lapse arguments
    parser.add_argument('--rate', type=int, default=1, required=False,
                        help='Iterations to run per frame.')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    gs = globals()
    simple_run(gs[args.network], gs[args.operation], gs[args.format], args)
