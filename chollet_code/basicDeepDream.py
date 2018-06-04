from keras.applications import inception_v3
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import sys
import scipy
from IPython.display import display
from PIL import Image

def check_args():
    if len(sys.argv) != 2:
        print("Usage: python3 basicDeepDream <path_to_image>")
        sys.exit()

def eval_loss_and_grads(x):
   outs = fetch_loss_and_grads([x])
   loss_value = outs[0]
   grad_values = outs[1]
   return loss_value, grad_values

# This function runs gradient ascent for a number of iterations
def gradient_ascent(x, iterations, step, max_loss=None):
   for i in range(iterations):
       loss_value, grad_values = eval_loss_and_grads(x)
       if max_loss is not None and loss_value > max_loss:
           break
       print('...Loss value at', i, ':', loss_value)
       x += step * grad_values
   return x

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

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

check_args()

K.set_learning_phase(0)

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

layer_contributions = {'mixed2': 0.2,
                       'mixed3': 3.,
                       'mixed4': 4.,
                       'mixed5': 1.5,
                       }

layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.)

for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))

    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling

# This tensor hold the generated image
dream = model.input

# Computes the gradients of the dream with regard to the loss
grads = K.gradients(loss, dream)[0]

# Normalizes the gradients
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# Sets up a Keras function to retrieve the value of the loss and gradients,
# given an input image
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)



step = 0.01 # Gradient ascent step size
num_octave = 3 # Number of scales at which to run the gradient ascent
octave_scale = 1.4 # Size ratio between the scales
iterations = 20 # Number of ascent steps to run at each scale
# If the loss grows larger than 10, you'll interrupt the gradient-ascent
# process to avoid ugly artifacts
max_loss = 10.

# Fill this with the path to the image you want to use
base_image_path = sys.argv[1]

# FIX THIS
img = preprocess_image(base_image_path)

# Prepares a list of shape tuples defining the different scales at which to run
# gradient ascent
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i))
        for dim in original_shape])
    successive_shapes.append(shape)

# Reverses the list of shapes so they're in increasing order
successive_shapes = successive_shapes[::-1]

#Resizes the Numpy array of the image to the smallest scale
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    # Scale up the dream image
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)

    # Computes the high-quality version of the original image at this size
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    # The difference between the two is the detail that was lost when scaling up
    lost_detail = (same_size_original - upscaled_shrunk_original_img)

    # Reinjects lost detail into the dream
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')

save_img(img, fname='final_dream.png')
