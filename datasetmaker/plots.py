import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2

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

def plot_shadow_positions(image, shades, ellipses):
    colors = [(255,0,0), (0,255,0), (0,0,255), 
              (255,0,255), (255,255,0), (255,0,255), 
              (128,0,0), (0,128,0), (0,0,128),
              (128,0,128), (128,128,0),(128,0,128)]
    
    for index, shadow in enumerate(shades):
        color = colors[index]
        image = cv2.ellipse(image, ellipses[index].center, ellipses[index].radius, ellipses[index].angle, 0, 360, color, 5)

        for point in shadow:
            image = cv2.circle(image, point, 10, color, -1)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.axis('off')
    ax.imshow(image)
    return fig

def plot_sample_positions(image, groups):
    colors = [(255,0,0), (0,255,0), (0,0,255), 
              (255,0,255), (255,255,0), (255,0,255), 
              (128,0,0), (0,128,0), (0,0,128),
              (128,0,128), (128,128,0),(128,0,128),
              (64,0,0), (0,64,0),(0,0,64),
              (64,0,64), (64,64,0),(64,0,64)]
    
    for index, points in enumerate(groups):
        color = colors[index]

        for point in points:
            image = cv2.circle(image, point, 10, color, -1)

    fig, ax = plt.subplots(figsize=(20,10))
    ax.axis('off')
    ax.imshow(image)
    return fig