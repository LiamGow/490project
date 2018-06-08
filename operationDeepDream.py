import numpy as np
from keras import backend as K

from myLib import Operation


class DeepDream(Operation):

    def __init__(self, args, network):
        self.args = args
        model = network().model
        self.layer_dict = dict([(layer.name, layer) for layer in model.layers])
        self.flgrads = self.get_flgrad(self.get_loss(), model.input)

    def apply(self, img, iterations):
        return self.gradient_ascent(img,
                                    iterations=iterations,
                                    step=self.args.step,
                                    max_loss=self.args.max_loss)

    @staticmethod
    def get_flgrad(loss, dream):

        grads = K.gradients(loss, dream)[0]
        grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  # normalize
        outputs = [loss, grads]

        return K.function([dream], outputs)

    def gradient_ascent(self, x, iterations, step, max_loss=None):
        for i in range(iterations):
            loss_value, grad_values = self.eval_loss_and_grads(x)
            if max_loss and loss_value > max_loss:
                break
            x += step * grad_values
        return x

    # deep dream functions
    def eval_loss_and_grads(self, x):
        outs = self.flgrads([x])
        loss_value = outs[0]
        grad_values = outs[1]
        return loss_value, grad_values

    def get_loss(self):
        lyr_contr = self.get_lyr_contribs(depth_tgt=self.args.abstraction)

        loss = K.variable(0.)
        for layer_name in lyr_contr:
            coeff = lyr_contr[layer_name]
            print("Layer", layer_name, "contributing", coeff)
            activation = self.layer_dict[layer_name].output
            scaling = K.prod(K.cast(K.shape(activation), 'float32'))
            lyrloss = (coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :]))
                       / scaling)
            loss = loss + lyrloss

        return loss

    def get_lyr_contribs(self, depth_tgt=0.5):
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
        for lyrnm, lyr in self.layer_dict.items():
            if filter_str in lyrnm:
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
