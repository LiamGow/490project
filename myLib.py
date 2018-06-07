class Cfg:
    src_dir = "./imsrc/"
    out_dir = "./imout/"
    abstraction_lvl = 1
    max_loss = None
    grad_step = 0.01
    iterct = 25

    img_nm = None
    img_nm_base = None
    nframes = None


class Network:

    def __init__(self):
        self.model = None

    @staticmethod
    def preprocess(img):
        pass

    @staticmethod
    def postprocess(img):
        pass


class Operation:

    def apply(self, img, iterations):
        pass


class Output:

    def load(self):
        pass

    def loop_cond(self):
        pass
