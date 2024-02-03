import matplotlib.pyplot as plt
from PIL import Image
import random

def plot_samples(images, masks):
    columns = 4
    rows = 4

    fig, axs = plt.subplots(rows, columns)
    fig.patch.set_facecolor('xkcd:black')
    flatten_axs = axs.flat
    for i in range(len(flatten_axs) - 1):
        index = random.randrange(0, len(images) - 1)
        flatten_axs[i].axis('off')
        flatten_axs[i+1].axis('off')

        if i % 2 == 0:
            image = Image.open(images[index])
            mask = Image.open(masks[index])
            flatten_axs[i].imshow(image)
            flatten_axs[i+1].imshow(mask)

    return fig

def plot_corns(images: list):
    fig, axs = plt.subplots(1,5)
    fig.patch.set_facecolor('xkcd:black')
    for ax in axs:
        ax.axis('off')
        index = random.randint(0, len(images) - 1)
        image = Image.open(images[index])
        ax.imshow(image)
    
    plt.tight_layout()

    return fig