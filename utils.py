import numpy as np
import matplotlib.pyplot as plt
import indices
import cv2
import os

kernel_shapes = {
    "Rectangular": cv2.MORPH_RECT,
    "Elliptical": cv2.MORPH_ELLIPSE,
    "Cross-shaped": cv2.MORPH_CROSS
}

def get_gray_image(rgb_image, vegetative_index):
    if vegetative_index == 0:
        gray_image = indices.get_exg(rgb_image)
    elif vegetative_index == 1:
        gray_image = indices.get_exr(rgb_image)
    elif vegetative_index == 2:
        gray_image = indices.get_ndi(rgb_image)
    elif vegetative_index == 3:
        gray_image = indices.get_cive(rgb_image)
    elif vegetative_index == 4:
        gray_image = indices.get_exgr(rgb_image)
    elif vegetative_index == 5:
        gray_image = indices.get_veg(rgb_image)
    else:
        gray_image = indices.get_com(rgb_image)
    return gray_image

def get_suggested_threshold(gray_image):
    th, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def build_mask(th, gray_image, apply_erosion=False, 
                apply_dilation=False, erosion_kernel=None, dilation_kernel=None, 
                erosion_kernel_shape=None, dilation_kernel_shape=None):
    mask = np.zeros_like(gray_image)
    mask[gray_image < th] = 255

    if apply_erosion and erosion_kernel is not None and erosion_kernel_shape is not None:
        kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8) 
        mask = cv2.erode(mask, kernel, erosion_kernel_shape)

    if apply_dilation and dilation_kernel is not None and dilation_kernel_shape is not None:
        kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8) 
        mask = cv2.dilate(mask, kernel, dilation_kernel_shape)

    return mask

def get_image_hist(gray_image):
    hist, _ = np.histogram(gray_image, bins=256, range=(0, 255))
    return hist

def plot_image(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    return fig

def get_color_mask(mask, random_color=False):
    color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
    replacement_color = [255, 0, 200, 80]
    color_mask[mask == 255] = replacement_color
    color_mask[mask != 255] = [0,0,0,0]
    return color_mask

def get_colorful_mask(mask, color):
    color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
    replacement_color = [255, 0, 200, 80]
    color_mask[mask == 255] = replacement_color
    color_mask[mask != 255] = [0,0,0,0]
    return color_mask

def search_image(path, image_paths=[]):
    image_extensions = ['jpg', 'jpeg']
    
    if os.path.exists(path) and os.path.isdir(path):
        for file in os.listdir(path):
            file_extension = file.split(".")[-1].lower()
            if file_extension in image_extensions:
                image_paths.append(os.path.join(path, file))
            elif os.path.isdir(os.path.join(path, file)):
                search_image(os.path.join(path, file), image_paths)
    return image_paths

def generate_img_fig(image, cmap='viridis', axis_on=False):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap)
    if not axis_on:
        ax.axis('off')
    return fig

def generate_line_plot(data, axis_on=True):
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(data)
    if not axis_on:
        ax.axis('off')

    return fig

def generate_img_with_mask_fig(image, mask, axis_on=False):
    color_maks = get_color_mask(mask)

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(color_maks)
    if not axis_on:
        ax.axis('off')

    return fig