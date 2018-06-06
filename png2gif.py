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
    img_path = Cfg.src_dir + Cfg.img_nm
    img_orig = preprocess_image(img_path)
    img = np.copy(img_orig)

    # deep dream algorithm
    gif_out_frames = []
    while (Cfg.nframes > 0):
        img = gradient_ascent(img, iterations=1, step=Cfg.grad_step,
                              max_loss=Cfg.max_loss)
        gif_out_frames.append(np.copy(img))
        Cfg.nframes -= 1

    # write out the dream sequence as gif
    gif_out_name = Cfg.img_nm_base + "_dreaming.gif"
    save_gif(gif_out_frames, gif_out_name)


def check_args():
    if len(sys.argv) != 3:
        print("Usage: python3 png2gif.py <src_img> <#frames>")
        sys.exit()
    Cfg.img_nm = sys.argv[1]
    Cfg.img_nm_base = Cfg.img_nm[0:Cfg.img_nm.find('.')]
    Cfg.nframes = int(sys.argv[2])

if __name__ == "__main__":
    main()
