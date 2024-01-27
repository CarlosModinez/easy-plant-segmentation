import os

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
    families = os.listdir(dataset_path)
    return _clear_list(families)

def get_weed_species(dataset_path: str, families: list) -> list:
    species = []
    for family in families:
        _species = os.listdir(os.path.join(dataset_path, family))
        _species = _clear_list(_species)
        species.extend(_species)
    return species

def get_weeds_images_and_masks(dataset_path: str, families: list, species: list) -> (list, list):
    images = []
    masks = []
    for family in families:
        all_species = os.listdir(os.path.join(dataset_path, family))
        for specie in all_species:
            if specie in species:
                for subfile in filter(lambda subfile: "." not in subfile, os.listdir(os.path.join(os.path.join(dataset_path, family), specie))):
                    image_names = os.listdir(os.path.join(os.path.join(os.path.join(dataset_path, family), specie), subfile))
                    for image_name in image_names:
                        if 'JPG' in image_name:
                            images.append(os.path.join(os.path.join(os.path.join(os.path.join(dataset_path, family), specie), subfile), image_name))
                        elif 'png' in image_name:
                            masks.append(os.path.join(os.path.join(os.path.join(os.path.join(dataset_path, family), specie), subfile), image_name))

    return (images, masks)