import numpy as np
import cv2

def _add_padding(image, required_x, required_y):
    h, w = image.shape[0], image.shape[1]
    pad_y, pad_x = int(required_y), int(required_x)
    padded_image_h, padded_image_w = h+pad_y*2, w+pad_x*2

    if len(image.shape) == 2:
        padded_image = np.zeros((padded_image_h, padded_image_w), dtype=np.uint8)
        padded_image[pad_y:h+pad_y, pad_x:w+pad_x] = image
        return padded_image
    elif len(image.shape) == 3:
        padded_image = np.zeros((padded_image_h, padded_image_w, image.shape[-1]), dtype=np.uint8)
        padded_image[pad_y:h+pad_y, pad_x:w+pad_x, :] = image
        return padded_image
    else:
        return None

def crop_image(image: np.array, crop_size: np.array):
    crops = []
    output_crop_size = (224, 224)
    image_h, image_w = image.shape[0], image.shape[1]
    crop_h, crop_w = crop_size

    n_crops_x = int(np.ceil(image_w / crop_w))
    n_crops_y = int(np.ceil(image_h / crop_h))

    required_x = ((n_crops_x * crop_w) - image_w)/2
    required_y = ((n_crops_y * crop_h) - image_h)/2

    padded_image = _add_padding(image, required_x, required_y)

    for y in range(0, n_crops_y):
        for x in range(0, n_crops_x):
            if len(image.shape) == 2:
                crop = padded_image[y*crop_h:y*crop_h+crop_h, x*crop_w:x*crop_w+crop_w]
                crop = cv2.resize(crop, output_crop_size, cv2.INTER_AREA)
                crops.append(crop)
            elif len(image.shape) == 3:
                crop = padded_image[y*crop_h:y*crop_h+crop_h, x*crop_w:x*crop_w+crop_w, :]
                crop = cv2.resize(crop, output_crop_size, cv2.INTER_AREA)
                crops.append(crop)
    return crops, n_crops_x, n_crops_y