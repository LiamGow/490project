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

def set_contributions(TODO_some_default_params=0): #TODO make more easily configurable
    return {'mixed2': 0.2,
            'mixed3': 3.,
            'mixed4': 4.,
            'mixed5': 1.5,
            }
    
def main():
    global fetch_loss_and_grads
    check_args()

    K.set_learning_phase(0)  # disables all training specific operations
    model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
    #print(model.summary())

    layer_contributions = set_contributions()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    loss = K.variable(0.)
    for layer_name in layer_contributions:
        print(layer_name)
        coeff = layer_contributions[layer_name]
        activation = layer_dict[layer_name].output
        scaling = K.prod(K.cast(K.shape(activation), 'float32'))
        th = coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling
        loss += th

    dream = model.input
    grads = K.gradients(loss, dream)[0]
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  # normalize
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)

# begin deep dream algorithm
    #hyperparams
    step = 0.01
    nframes = int(sys.argv[2])
    max_loss = None

    #set up path
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
        #print('...Loss value at', i, ':', loss_value)
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


if __name__ == "__main__":
    main()
