from keras.applications import inception_v3
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import sys
import scipy
from IPython.display import display
from PIL import Image
import imageio

BASE_SRC = "./imsrc/"
BASE_OUT = "./imout/"
fetch_loss_and_grads = None

def set_contributions(arch_dict, depth_tgt=0.5):
    # depth_tgt is the targeted level of abstraction used to determine which
    # layers influence the final loss function most. layers nearer to the end
    # of the architecture should produce representations with higher levels of
    # abstraction, for example, an animal eye is a higher layer of abstraction
    # than a geometric pattern- layers nearer to the beginning of the
    # architecture will produce more geometric-pattern-like representations.
    #
    # depth_tgt of 0 will result in a look closer to the geometric-pattern end
    # of the spectrum, while a depth_tgt of 1 will result in a look in which
    # higher levels of abstraction should become visible.
    #
    # chollet_base = {'mixed2': 0.2, 'mixed3': 3., 'mixed4': 4., 'mixed5': 1.5}

    # list of chosen layers to be used in loss fn, sorted
    # in order of appearance in the architecture (or close to it)
    candidates = []

    filter_str = "mixed"
    for lyrnm, lyr in arch_dict.items():
        if (filter_str in lyrnm):
            non_filter_part = lyrnm[len(filter_str):]

            try:
                srtV = int(non_filter_part)
            except ValueError:
                srtV = int(non_filter_part[:non_filter_part.find('_')])

            candidates.append((srtV, lyrnm))

    tgt_num_lyrs_that_influence_loss = 5

    candidates.sort()
    cand_depths = np.linspace(0, 1, len(candidates))
    err = abs(cand_depths - depth_tgt)
    coeffs = -np.log(err + 0.01)
    prob = coeffs / np.sum(coeffs)  # adjust prob so its a distr that sums to 1
    selections = np.random.multinomial(tgt_num_lyrs_that_influence_loss, prob)

    rtn = {}
    for i, result in enumerate(selections):
        if (result):
            rtn[candidates[i][1]] = result * coeffs[i]

    return rtn


def set_FLGRAD(loss, dream):
    global fetch_loss_and_grads

    grads = K.gradients(loss, dream)[0]
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  # normalize
    outputs = [loss, grads]

    fetch_loss_and_grads = K.function([dream], outputs)


def main():
    check_args()
    K.set_learning_phase(0)  # disables all training specific operations
    model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_contributions = set_contributions(layer_dict, depth_tgt=0.99)

    loss = K.variable(0.)
    for layer_name in layer_contributions:
        print(layer_name)
        coeff = layer_contributions[layer_name]
        activation = layer_dict[layer_name].output
        scaling = K.prod(K.cast(K.shape(activation), 'float32'))
        lyrloss = (coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :]))
                   / scaling)
        loss = loss + lyrloss

    set_FLGRAD(loss, model.input)


# begin deep dream algorithm
    # hyperparams
    step = 0.01
    nframes = int(sys.argv[2])
    max_loss = None

    # set up path
    img_name = sys.argv[1]
    path = BASE_SRC + img_name
    img_base_name = img_name[0:img_name.find('.')]

    img_orig = preprocess_image(path)
    img = np.copy(img_orig)

    gif_src_frames = []
    while (nframes > 0):
        img = gradient_ascent(img, iterations=1, step=step, max_loss=max_loss)
        gif_src_frames.append(np.copy(img))
        nframes -= 1

    gif_out_name = img_base_name + "_dreaming.gif"
    save_gif(gif_src_frames, gif_out_name)


# --------------------- helper function section -------------------
def check_args():
    if len(sys.argv) != 3:
        print("Usage: python3 gifDeepDream <src_img> <#frames>")
        sys.exit()


# deep dream functions
def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        x += step * grad_values
    return x


# img processing functions
def save_gif(frame_list, fname):
    print(fname)
    images = []
    filenames = []
    for i, frame in enumerate(frame_list):
        filename = "gif_frames/" + "frame" + str(i) + ".png"
        save_img(frame, filename)
        filenames.append(filename)
    for filename in filenames:
        images.append(imageio.imread(BASE_OUT + filename))
    imageio.mimsave(BASE_OUT + fname, images, duration=0.1)


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(BASE_OUT + fname, pil_img)


def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# ----------------------- run main ------------------------------
if __name__ == "__main__":
    main()
