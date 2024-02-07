import streamlit as st
import matplotlib.pyplot as plt
import datasetmaker.files as files
import datasetmaker.cropper as cropper
from sklearn.model_selection import train_test_split
import random
import os
import cv2
import numpy as np

########## constants ##########

ORIGINAL_DATASET_PATH = "/Users/carlosmodinez/Documents/TCC/brazilian_weeds_2020"

########## page config ##########

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

########## sidebar elements ##########

st.sidebar.title("Easy plant segmentation 🌱")
st.sidebar.caption("")
st.sidebar.markdown("---")

crop_height = int(st.sidebar.number_input("Crop height", value=1000, help="Crop height"))
crop_width = int(st.sidebar.number_input("Crop width", value=1000, help="Crop width"))

########## body elements ##########

def search_images(path, image_paths=[], extensions=['jpg', 'jpeg']):
    image_extensions = extensions

    if os.path.exists(path) and os.path.isdir(path):
        for file in os.listdir(path):
            file_extension = file.split(".")[-1].lower()
            if file_extension in image_extensions:
                image_paths.append(os.path.join(path, file))
            elif os.path.isdir(os.path.join(path, file)):
                search_images(os.path.join(path, file), image_paths)
    return image_paths

def find_class_index(image_path: str):
    classes = [
        "Amaranthus hybridos", "Cyclospermum leptophyllum", "Tridax procumbens",
        "Sonchus oleraceus", "Bidens-pilosa", "Lelidium virginum", "Ipomoeae purpurea",
        "Leonurus sibiricus", "Sorghum arundinaceum", "Rhynchelytrumrepens",
        "Digitaria insularis", "Choris barbata", "Cenchus echinatus", "Physalis angulata"
    ]

    for index, class_name in enumerate(classes):
        if class_name.lower() in image_path.lower():
            return index
    return None

def build_dataset():
    destination_seg_dataset = f"/Users/carlosmodinez/Documents/TCC/datasets/{crop_height}x{crop_width}"
    image_paths = search_images(ORIGINAL_DATASET_PATH, [])

    train, test_val = train_test_split(image_paths, test_size=0.3, random_state=42)
    test, val = train_test_split(test_val, test_size=0.5, random_state=42)

    for split in [train, test, val]:
        if split == train:
            destination_images_path =  os.path.join(destination_seg_dataset, 'train/images')
            destination_masks_path =  os.path.join(destination_seg_dataset, 'train/masks')

            os.makedirs(destination_images_path)
            os.makedirs(destination_masks_path)
        elif split == test:
            destination_images_path =  os.path.join(destination_seg_dataset, 'test/images')
            destination_masks_path =  os.path.join(destination_seg_dataset, 'test/masks')
            
            os.makedirs(destination_images_path)
            os.makedirs(destination_masks_path)
        else:
            destination_images_path =  os.path.join(destination_seg_dataset, 'val/images')
            destination_masks_path =  os.path.join(destination_seg_dataset, 'val/masks')

            os.makedirs(destination_images_path)
            os.makedirs(destination_masks_path)

        for index, image_path in enumerate(split):
            class_index = find_class_index(image_path)
            image = files.read_colorful_image(image_path)
            mask = cv2.imread(image_path.replace('JPG', 'png'), cv2.IMREAD_GRAYSCALE)

            im_crops, _, _ = cropper.crop_image(image, (crop_height, crop_width))
            mask_crops, _, _ = cropper.crop_image(mask, (crop_height, crop_width))

            for crop_index in range(len(im_crops)):
                destination_image = os.path.join(destination_images_path, f"{index}_{crop_index}.png")
                destination_mask = os.path.join(destination_masks_path, f"{index}_{crop_index}.png")

                mask_crop = mask_crops[crop_index]
                mask_crop[mask_crop != 0] = 255 # to remove the effect of resizing

                modified_mask = np.zeros((mask_crop.shape[0], mask_crop.shape[1], 1), dtype=np.uint8) #(224 x 224) -> (224 x 224 x 1)
                modified_mask[:, :, 0] = ((mask_crop/255) * (class_index + 1)).astype('uint8') # (0 > 255) -> (0 > class_index)

                files.save_image(im_crops[crop_index], destination_image)
                files.save_image(modified_mask, destination_mask)

def show_example():
    image_paths = search_images(ORIGINAL_DATASET_PATH, [])
    ex_index = random.randint(0, len(image_paths) - 1)
    image_path = image_paths[ex_index]
    ex_image = files.read_colorful_image(image_path)
    ex_mask = files.read_grayscale_image(image_path.replace('JPG', 'png'))

    st.image(ex_image)
    st.image(ex_mask)

    img_size = (crop_height, crop_width)
    im_crops, n_crops_x, n_crops_y = cropper.crop_image(ex_image, img_size)
    mask_crops, _, _ = cropper.crop_image(ex_mask, img_size)

    img_fig, im_axs = plt.subplots(n_crops_y, n_crops_x)
    mask_fig, mask_axs = plt.subplots(n_crops_y, n_crops_x)

    im_axs = im_axs.flatten()
    mask_axs = mask_axs.flatten()

    for index in range(len(im_axs)):
        im_axs[index].axis('off')
        mask_axs[index].axis('off')
        im_axs[index].imshow(im_crops[index])
        mask_axs[index].imshow(mask_crops[index])

    st.pyplot(img_fig)
    st.pyplot(mask_fig)

st.sidebar.button("Generate images", on_click=build_dataset)
st.button("Example", on_click=show_example)
st.sidebar.button("Update example", on_click=show_example)