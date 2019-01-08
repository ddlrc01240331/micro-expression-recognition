import matplotlib.pyplot as plt

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
