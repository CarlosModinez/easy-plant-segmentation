import streamlit as st
import os
import datasetmaker.files as files
import datasetmaker.plots as plots

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
    st.pyplot(plots.plot_corns(corn_images))

families = files.get_weed_families(weeds_dataset_path)
selected_families = st.multiselect("Família da daninha", families)

species = files.get_weed_species(weeds_dataset_path, selected_families)
selected_species = st.multiselect("Família da daninha", species)

images, masks = files.get_weeds_images_and_masks(weeds_dataset_path, selected_families, selected_species)

if len(images) > 0:
    fig = plots.plot_samples(images, masks)
    st.pyplot(fig)