from myLib import *
from dreamLib import *
from imgLib import *

from keras.applications import inception_v3

def main():
    check_args()
    K.set_learning_phase(0)  # disables all training specific operations
    model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    loss = get_loss(layer_dict)
    set_FLGRAD(loss, model.input)

    # get the input img
    gif_path = Cfg.src_dir + Cfg.img_nm
    gif_src_frames = preprocess_gif(gif_path)
    gif_out_frames = []

    # deep dream algorithm
    print("Total frames", len(gif_src_frames))
    for i, frame in enumerate(gif_src_frames):
        print("Processing frame ", i)
        frm = gradient_ascent(frame, iterations=Cfg.iterct,
                              step=Cfg.grad_step,
                              max_loss=Cfg.max_loss)

        gif_out_frames.append(np.copy(frm))


    # write out the dream sequence as gif
    gif_out_name = Cfg.img_nm_base + "_dream.gif"
    save_gif(gif_src_frames, gif_out_name)

def check_args():
    if len(sys.argv) != 2:
        print("Usage: python3 gif2gif.py <src_gif>")
        sys.exit()
    Cfg.img_nm = sys.argv[1]
    Cfg.img_nm_base = Cfg.img_nm[0:Cfg.img_nm.find('.')]


if __name__ == "__main__":
    main()
