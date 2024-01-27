import numpy as np
import random
from PIL import Image
from skimage.transform import resize, rotate

def generate_weeds(density: int, images: list, masks: list, corns: list, corns_mask) -> (np.array, np.array, np.array):
    # there are corn images and weed images
    corn_index = random.randint(0, len(corns) - 1)
    corn_image_path = corns[corn_index]
    corn_mask_path = corns_mask[corn_index]
    
    corn_image = np.array(Image.open(corn_image_path))
    corn_mask = np.array(Image.open(corn_mask_path))
    weeds = np.zeros_like(corn_image)
    img_h, img_w, _ = corn_image.shape

    for _ in range(0, density):
        image_index = random.randint(0, len(images) - 1)
        image_path = images[image_index]
        mask_path = masks[image_index]
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))

        image[mask == 0] = 0
        resize_factor = random.randint(5, 50) / 1000 # image betwwen 20% and 40% of its size
        new_size = tuple(int(size * resize_factor) for size in image.shape[:2])
        angle = random.random() * 360

        image = resize(image, new_size)
        image = rotate(image, angle)
        image = (image * 255).astype(np.uint8)

        x_offset = random.randint(0, img_w - image.shape[1])
        y_offset = random.randint(0, img_h - image.shape[0])

        weeds[y_offset: y_offset + image.shape[0], x_offset: x_offset + image.shape[1], :][~np.all(image == [0,0,0], axis=-1)] = image[~np.all(image == [0,0,0], axis=-1)]
    return corn_image, corn_mask[:, :, 3], weeds

def calculate_histogram(image):
    histogram = np.zeros(256, dtype=int)
    for pixel in image.ravel():
        if pixel != 0:
            histogram[pixel] += 1
    return histogram

def calculate_cdf(histogram):
    cdf = np.cumsum(histogram)
    return cdf

def histogram_equalization(image):
    histogram = calculate_histogram(image)
    cdf = calculate_cdf(histogram)

    num_pixels = image.size
    image_equalized = (cdf[image] * 255.0 / num_pixels).astype(np.uint8)
    
    return image_equalized

def match_histogram(source, target):
    source_histogram = calculate_histogram(source)
    source_cdf = calculate_cdf(source_histogram)
    
    target_histogram = calculate_histogram(target)
    target_cdf = calculate_cdf(target_histogram)

    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapping[i] = np.argmin(np.abs(source_cdf[i] - target_cdf))
    
    matched_image = mapping[source]
    
    return matched_image

def split_RGB(image):
    return image[:,:,0], image[:,:,1], image[:,:,2] 

def match_rgb_histogram(source, target):
    source_R, source_G, source_B = split_RGB(source)
    target_R, target_G, target_B = split_RGB(target)

    matched_R = match_histogram(source_R, target_R)
    matched_G = match_histogram(source_G, target_G)
    matched_B = match_histogram(source_B, target_B)

    return np.dstack((matched_R, matched_G, matched_B))

