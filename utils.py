import torch
import numpy as np
import matplotlib.pyplot as plt

def get_accuracy(y_pred, y_target):
    n_correct = torch.eq(y_pred, y_target).sum().item()
    accuracy = n_correct / len(y_pred) * 100
    return accuracy


def imshow(image_show, title=None):
    # Change C x H x W into H x W x C to show the image(plt)
    image_show = image_show.numpy().transpose((1, 2, 0))
    # Re-Normalize the imgage
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_show = std * image_show + mean
    # Limit the value into approperiate interval
    # RGB image float (0, 1) and integers (0, 255)
    image_show = np.clip(image_show, 0, 1)

    plt.imshow(image_show)

    # Show the label of each images in the grid
    if title is not None:
        plt.title(title)
    plt.pause(0.001)