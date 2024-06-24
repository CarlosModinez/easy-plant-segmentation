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
st.sidebar.markdown("Made by [Carlos Modinez](https://www.linkedin.com/in/carlos-modinez/)")
st.sidebar.caption("")
st.sidebar.markdown("---")

crop_height = int(st.sidebar.number_input("Crop height", value=256, help="Crop height"))
crop_width = int(st.sidebar.number_input("Crop width", value=256, help="Crop width"))

st.sidebar.markdown("Padding size")

padding_width = int(st.sidebar.number_input("Padding width", value=30))
padding_height = int(st.sidebar.number_input("Padding Height", value=30))

st.sidebar.markdown(f"Your output image size is {crop_height + padding_height}:{crop_width + padding_width}")

########## body elements ##########
images_path = st.empty().text_input("Images folder", help='All image from this folder will be loaded', key="image_url")
annot_path = st.empty().text_input("Annotations folder")

images = [os.path.join(images_path, f) for f in os.listdir(images_path) if 'png' in f]
annotations = [os.path.join(images_path, f) for f in os.listdir(images_path) if 'png' in f]
index = 0

if len(images) > 0:
    image_path = images[index]
    image = cv2.imread(images[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig_1, ax_1 = plt.subplots()
    ax_1.imshow(image)
    ax_1.axis('off')
    st.pyplot(fig_1)

    crops, n_crops_x, n_crops_y = cropper.crop_image(image, (crop_height, crop_width), (padding_width, padding_height))
    fig, axes = plt.subplots(n_crops_y, n_crops_x, figsize=(n_crops_x*10, n_crops_y * 10))
    st.markdown(f'crops: {len(crops)}; n_crops: {n_crops_x}:{n_crops_y} ; axes len: {len(axes)}')
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(crops[i])
        ax.axis('off')

    st.pyplot(fig)