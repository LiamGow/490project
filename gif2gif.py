from myLib import *
from imgLib import *

from dreamLib import DeepDream
from inceptionV3Lib import InceptionV3

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



# def main():
#     check_args()
#     K.set_learning_phase(0)  # disables all training specific operations
#     model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
#     layer_dict = dict([(layer.name, layer) for layer in model.layers])
#
#     loss = get_loss(layer_dict)
#     set_FLGRAD(loss, model.input)
#
#     # get the input img
#     gif_path = Cfg.src_dir + Cfg.img_nm
#     gif_src_frames = preprocess_gif(gif_path)
#     gif_out_frames = []
#
#     # deep dream algorithm
#     print("Total frames", len(gif_src_frames))
#     for i, frame in enumerate(gif_src_frames):
#         print("Processing frame ", i)
#         frm = gradient_ascent(frame, iterations=Cfg.iterct,
#                               step=Cfg.grad_step,
#                               max_loss=Cfg.max_loss)
#
#         gif_out_frames.append(np.copy(frm))
#
#
#     # write out the dream sequence as gif
#     save_gif(Cfg.img_nm_base + "_dream.gif", gif_src_frames)

def check_args():
    if len(sys.argv) != 2:
        print("Usage: python3 gif2gif.py <src_gif>")
        sys.exit()
    Cfg.img_nm = sys.argv[1]
    Cfg.img_nm_base = Cfg.img_nm[0:Cfg.img_nm.find('.')]
