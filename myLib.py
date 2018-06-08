# class Cfg:
#     src_dir = "./imsrc/"
#     out_dir = "./imout/"
#     abstraction_lvl = 1
#     max_loss = None
#     grad_step = 0.01
#
#     image_path = None
#     image_base = None
#     style_path = None
#     iterations = None
#     rate = 1


class Network:

    def __init__(self):
        self.model = None

    @staticmethod
    def preprocess_image(img):
        return img

    @staticmethod
    def deprocess_image(img):
        return img


class Operation:

    def __init__(self, model):
        pass

    def apply(self, img, iterations):
        pass


class Output:

    def __init__(self):
        self.load = None

    def run(self, img, operation):
        pass
