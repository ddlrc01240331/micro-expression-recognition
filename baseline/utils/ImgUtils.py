import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.figure()
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

class CropFromHead(object):

    def __init__(self, height):
        self.height = height

    def __call__(self, sample):
        w, h= sample.size
        new_img = sample.crop((0, 0, w, self.height))
        return new_img

class PaddingToSquare(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        w, h = sample.size
        assert h>w, 'casme has larger height than width'
        delta_l = int((h - w)/2)
        delta_r = int((h-w+1)/2)
        left, top, right, bottom = delta_l, 0, delta_r, 0
        new_im = ImageOps.expand(sample, (left, top, right, bottom))
        # print(new_im.size)
        return new_im
