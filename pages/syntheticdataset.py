import streamlit as st
import os
import datasetmaker.files as files
import datasetmaker.plots as plots
import datasetmaker.cornweed as cornweed
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

cornspacing_dataset_path = "/Users/carlosmodinez/Documents/TCC/cornspacing/cornspacing-dataset"
weeds_dataset_path = "/Users/carlosmodinez/Documents/TCC/brazilian_weeds_2020"

tillage_type = os.listdir(cornspacing_dataset_path)
tillage_type = filter(lambda type: "." not in type, tillage_type)
tillages_selected = st.multiselect("Tipo de lavoura", tillage_type)

if len(tillages_selected) > 0:
    corn_images, corn_masks = files.get_corn_images(cornspacing_dataset_path, tillages_selected)
    # st.pyplot(plots.plot_corns(corn_images))

families = files.get_weed_families(weeds_dataset_path)
selected_families = st.multiselect("Família da daninha", families)

species = files.get_weed_species(weeds_dataset_path, selected_families)
selected_species = st.multiselect("Família da daninha", species)

images, masks = files.get_weeds_images_and_masks(weeds_dataset_path, selected_families, selected_species)

if len(images) > 0:
    # fig = plots.plot_samples(images, masks)
    # st.pyplot(fig)
    img = Image.open(images[0])
    mask = Image.open(masks[0])

if len(tillages_selected) > 0 and len(images) > 0:
    corns_img, corns_mask, weeds = cornweed.generate_weeds(5, images, masks, corn_images, corn_masks)
    masked_corns = corns_img.copy()
    masked_corns[corns_mask == 0] = 0
    matched_weeds = cornweed.match_rgb_histogram(weeds, masked_corns)
    
    mixed_img = corns_img.copy()
    mixed_img[~np.all(matched_weeds == [0,0,0], axis=-1)] = matched_weeds[~np.all(matched_weeds == [0,0,0], axis=-1)] # add weeds
    mixed_img[~np.all(masked_corns == [0,0,0], axis=-1)] = masked_corns[~np.all(masked_corns == [0,0,0], axis=-1)] # add corns over the weeds 

    fig, axs = plt.subplots(1, 4, figsize=(25,40))
    axs[0].imshow(corns_img)
    axs[1].imshow(masked_corns)
    axs[2].imshow(matched_weeds)
    axs[3].imshow(mixed_img)

    st.pyplot(fig)