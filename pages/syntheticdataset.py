import streamlit as st
import os
import datasetmaker.files as files
import datasetmaker.plots as plots
import datasetmaker.cornweed as cornweed
import matplotlib.pyplot as plt
import cv2
import numpy as np

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

cornspacing_dataset_path = "/Users/carlosmodinez/Documents/TCC/cornspacing/cornspacing-dataset"
weeds_dataset_path = "/Users/carlosmodinez/Documents/TCC/brazilian_weeds_2020"

tillage_type = os.listdir(cornspacing_dataset_path)
tillage_type = filter(lambda type: "." not in type, tillage_type)
tillages_selected = st.sidebar.multiselect("Tipo de lavoura", tillage_type)

if len(tillages_selected) > 0:
    corn_images, corn_masks = files.get_corn_images(cornspacing_dataset_path, tillages_selected)

families = files.get_weed_families(weeds_dataset_path)
selected_families = st.sidebar.multiselect("Família da daninha", families)

species = files.get_weed_species(weeds_dataset_path, selected_families)
selected_species = st.sidebar.multiselect("Espécies da daninha", species)

groups = []
if len(tillages_selected) > 0 and len(selected_families) > 0 and len(selected_species) > 0:
    generation_mode = st.sidebar.multiselect('Generation mode', ['Shades', 'Scattered'])
    
    corn_image = cv2.imread(corn_images[0])    
    groups = []

    if 'Shades' in generation_mode:
        density = int(st.sidebar.number_input('Fator de densidade', value=5, max_value=100, min_value=0))
        st.sidebar.markdown('Range do número de sombras')
        min_shadows = int(st.sidebar.number_input('Mínimo', value=1, max_value=10, min_value=1))
        max_shadows = int(st.sidebar.number_input('Máximo', value=min_shadows+2, max_value=10, min_value=min_shadows))

        st.sidebar.markdown('Atributos da ellipse (% \da largura da imagem)')
        min_radius = int(st.sidebar.number_input('Mínimo', value=12, max_value=1000, min_value=1)) / 100
        max_radius = int(st.sidebar.number_input('Máximo', value=100, max_value=1000, min_value=1)) / 100
        std_deviation = int(st.sidebar.number_input('Desvio padrão da distribuição de ervas', value=50, max_value=100, min_value=1, help='Desvio padrão da distribuição da sombra em função do tamanho do raio'))/100

        number_of_shadows = np.random.randint(min_shadows, max_shadows)

        shadow_groups, ellipses = cornweed.get_shadow_weed_positions(
            corn_image, 
            number_of_shadows=number_of_shadows, 
            density=density,
            std_deviation=std_deviation,
            min_radius=min_radius,
            max_radius=max_radius
        )

        groups.extend(shadow_groups)
        preview = plots.plot_shadow_positions(corn_image, groups, ellipses)
        st.pyplot(preview)

    if 'Scattered' in generation_mode:
        min_number_of_species = int(st.sidebar.number_input('Min number of species', value=1, max_value=len(selected_species), min_value=1))
        max_number_of_species = int(st.sidebar.number_input('Max number of species', value=1, max_value=len(selected_species), min_value=1))

        min_samples = int(st.sidebar.number_input('Min number of samples', value=10, max_value=100, min_value=1))
        max_samples = int(st.sidebar.number_input('Max number of samples', value=min_samples+1, max_value=100, min_value=min_samples))

        corn_image = cv2.imread(corn_images[0])
        number_of_species = np.random.randint(min_number_of_species, max_number_of_species) if len(selected_species) > 1 else 1
        n_samples = []
        for _ in range(number_of_species):
            n_samples.append(np.random.randint(min_samples, max_samples))

        scattered_groups = cornweed.get_sample_weed_positions(corn_image, number_of_species, n_samples)

        for index, _ in enumerate(zip(groups, scattered_groups)):
            groups[index].extend(scattered_groups[index])
            scattered_groups[index] = []

        groups.extend(scattered_groups)

        preview = plots.plot_sample_positions(corn_image, groups)
        st.pyplot(preview)

def build_high_fidelity_example():
    groups = []
    if 'Shades' in generation_mode:
        number_of_shadows = np.random.randint(min_shadows, max_shadows)
        shadow_groups, _ = cornweed.get_shadow_weed_positions(
            corn_image, 
            number_of_shadows=number_of_shadows, 
            density=density,
            std_deviation=std_deviation,
            min_radius=min_radius,
            max_radius=max_radius
        )

        groups.extend(shadow_groups)

    if 'Scattered' in generation_mode:
        number_of_species = np.random.randint(min_number_of_species, max_number_of_species) if len(selected_species) > 1 else 1
        n_samples = []
        for _ in range(number_of_species):
            n_samples.append(np.random.randint(min_samples, max_samples))

        scattered_groups = cornweed.get_sample_weed_positions(corn_image, number_of_species, n_samples)

        for index, _ in enumerate(zip(groups, scattered_groups)):
            groups[index].extend(scattered_groups[index])
            scattered_groups[index] = []

        groups.extend(scattered_groups)


    images, masks, annotations = files.get_weeds_images_and_masks(
        weeds_dataset_path, 
        selected_families, 
        selected_species
    )

    image, annotation = cornweed.generate_weeds(
        groups = groups,
        species=selected_species,
        images = images,
        masks = masks,
        annotations=annotations, 
        corns=corn_images,
        corn_masks=corn_masks
    )

    st.image(image)
    st.image(annotation)
    
    return image, annotation

path = int(st.sidebar.number_input("Pasta", value=1, min_value=1, max_value=20)) # delete it

def generate_twenty_samples():
    for index in 3000:
        image, annotation = build_high_fidelity_example()
        cv2.imwrite(f'/Users/carlosmodinez/Documents/TCC/corn_weeds/0{str(path)}/images/{index}.png', image)
        cv2.imwrite(f'/Users/carlosmodinez/Documents/TCC/corn_weeds/0{str(path)}/annotations/{index}.png', annotation)

st.sidebar.button(
    'Generate high fidelity example', 
    on_click=build_high_fidelity_example
)

st.sidebar.button(
    'Generate 20 images',
    on_click=generate_twenty_samples
)