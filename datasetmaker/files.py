import os
import cv2
import numpy as np

def _clear_list(list: list) -> list:
    items_to_remove = ['.DS_Store', '.git']
    for item in items_to_remove:
        if item in list:
            list.remove(item)
    return list

def get_corn_images(dataset_path: str, tillages: list) -> (list, list):
    image_paths = []
    mask_paths = []
    for tillage in tillages:
        img_files = os.listdir(os.path.join(dataset_path, tillage))
        images = list(filter(lambda x: '.jpg' in x, img_files))
        images = sorted(images)

        mask_files = os.listdir(os.path.join(os.path.join(dataset_path, tillage),  "data/mark"))
        masks = list(filter(lambda x: '.png' in x, mask_files))
        masks = sorted(masks)

        for i in range(len(images)):
            image_paths.append(os.path.join(os.path.join(dataset_path, tillage), images[i]))
            mask_paths.append(os.path.join(os.path.join(os.path.join(dataset_path, tillage), "data/mark"), masks[i]))

    return image_paths, mask_paths

def get_weed_families(dataset_path: str) -> list:
    image_paths = os.path.join(dataset_path, 'images')
    families = os.listdir(image_paths)
    return _clear_list(families)

def get_weed_species(dataset_path: str, families: list) -> list:
    image_paths = os.path.join(dataset_path, 'images')
    species = []
    for family in families:
        _species = os.listdir(os.path.join(image_paths, family))
        _species = _clear_list(_species)
        species.extend(_species)
    return species

def get_weeds_images_and_masks(dataset_path: str, families: list, species: list) -> (list, list):
    images = []
    masks = []
    annotations = []
    images_paths = os.path.join(dataset_path, 'images')
    masks_paths = os.path.join(dataset_path, 'masks')
    annot_paths = os.path.join(dataset_path, 'annotations')

    for family in families:
        all_species = os.listdir(os.path.join(images_paths, family))
        for specie in all_species:
            if specie in species:
                for subfile in filter(lambda subfile: "." not in subfile, os.listdir(os.path.join(os.path.join(images_paths, family), specie))):
                    image_names = os.listdir(os.path.join(os.path.join(os.path.join(images_paths, family), specie), subfile))
                    for image_name in image_names:
                        if 'JPG' in image_name or 'png' in image_name:
                            images.append(os.path.join(os.path.join(os.path.join(os.path.join(images_paths, family), specie), subfile), image_name))
                            masks.append(os.path.join(os.path.join(os.path.join(os.path.join(masks_paths, family), specie), subfile), image_name).replace('JPG', 'png'))
                            annotations.append(os.path.join(os.path.join(os.path.join(os.path.join(annot_paths, family), specie), subfile, image_name)).replace('JPG', 'png'))

    return (images, masks, annotations)

def read_colorful_image(path) -> np.array:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def read_grayscale_image(path) -> np.array:
    return cv2.imread(path, 0)

def save_image(image: np.array, path: str) -> None:
    new_iamge = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, new_iamge)