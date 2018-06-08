from tensorflow.python.framework.errors_impl import ResourceExhaustedError


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

    def __init__(self, network):
        pass

    def apply(self, img, iterations):
        pass


class Output:

    def __init__(self):
        self.load = None

    def run(self, img, operation):
        pass
