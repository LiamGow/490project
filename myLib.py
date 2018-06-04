import numpy as np
from keras import backend as K
import sys

class Cfg():
    src_dir = "./imsrc/"
    out_dir = "./imout/"
    abstraction_lvl = 1
    max_loss = None
    grad_step = 0.01
    iterct = 25

    img_nm = None
    img_nm_base = None
    nframes = None



def get_lyr_contribs(arch_dict, depth_tgt=0.5):
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


def get_loss(layer_dict):
    lyr_contr = get_lyr_contribs(layer_dict,
                                 depth_tgt=Cfg.abstraction_lvl)

    loss = K.variable(0.)
    for layer_name in lyr_contr:
        coeff = lyr_contr[layer_name]
        print("Layer", layer_name, "contributing", coeff)
        activation = layer_dict[layer_name].output
        scaling = K.prod(K.cast(K.shape(activation), 'float32'))
        lyrloss = (coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :]))
                   / scaling)
        loss = loss + lyrloss

    return loss
