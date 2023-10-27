import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import utils
from indices import vegetative_indices

########## sidebar elements ##########

st.sidebar.title("Easy plant segmentation 🌱")
st.sidebar.caption("")
st.sidebar.markdown("Made by [Carlos Modinez](https://www.linkedin.com/in/carlos-modinez/)")
st.sidebar.caption("")
st.sidebar.markdown("---")

vegetative_index = st.sidebar.selectbox("Índice vegetativo", vegetative_indices.keys(), help="Principais índices vegetativos encontrados na literatura.")

st.sidebar.markdown("---")

toggle_erosion = st.sidebar.checkbox("Apply erosion", value=True, help="")
erosion_kernel_size = int(st.sidebar.number_input("Erosion kernel size", value=5, help="size", disabled=not(bool(toggle_erosion))))
erosion_kernel_shape = st.sidebar.selectbox("Erosion kernel shape", utils.kernel_shapes.keys(), help="Kernel", disabled=not(bool(toggle_erosion)))

st.sidebar.markdown("---")

toggle_dilation = st.sidebar.checkbox("Apply dilation", value=True, help="")
dilation_kernel_size = int(st.sidebar.number_input("Dilation kernel size", value=5, help="size", disabled=not(bool(toggle_dilation))))
dilation_kernel_shape = st.sidebar.selectbox("Dilation kernel shape", utils.kernel_shapes.keys(), help="Kernel", disabled=not(bool(toggle_dilation)))

########## body elements ##########

dataset_path_text = st.empty()
dataset_path = dataset_path_text.text_input("Dataset path", help='All image from this folder will be loaded', key="image_url")

list = utils.search_image(dataset_path)
# st.subheader(f"Total of {len(list)} images found")

if 'idx' not in st.session_state:
    st.session_state.idx = 0

def change():
    st.session_state.idx += 1
    if st.session_state.idx >= len(list):
        st.session_state.idx = 0

def save():
    mask_saved = Image.fromarray(mask.astype(np.uint8))
    mask_saved.save(mask_path)
    np.save(numpy_arr_path, mask)
    st.session_state.idx += 1
    if st.session_state.idx >= len(list):
        st.session_state.idx = 0

if len(list) > 0:
    st.image(list[st.session_state.idx])
    mask_path = list[st.session_state.idx]
    mask_path = mask_path.replace('JPG', 'png')

    numpy_arr_path = list[st.session_state.idx]
    numpy_arr_path = numpy_arr_path.replace('JPG', 'npy')

    img = Image.open(list[st.session_state.idx])
    gray_image = utils.get_gray_image(img, vegetative_index=vegetative_indices[vegetative_index])
    threshold = utils.get_suggested_threshold(gray_image)
    number = st.slider("Pick a number", 0, 255, int(threshold))
    hist = utils.get_image_hist(gray_image)
    mask = utils.build_mask(
        number, gray_image, apply_erosion=toggle_erosion, apply_dilation=toggle_dilation,
        erosion_kernel=erosion_kernel_size, dilation_kernel=dilation_kernel_size, 
        erosion_kernel_shape=utils.kernel_shapes[erosion_kernel_shape], 
        dilation_kernel_shape=utils.kernel_shapes[dilation_kernel_shape])

    col1, col2 = st.columns([5,5])

    with col1:
        st.button('Save mask', on_click=save, help=f'it will be saved as {mask_path}')
    with col2:
        st.button('Next image', on_click=change)

    with st.expander("🖼️ Original image", expanded=True):
        st.pyplot(utils.generate_img_with_mask_fig(img, mask))

    with st.expander("🎭 Mask", expanded=True):
        st.pyplot(utils.generate_img_fig(mask, cmap='gray'))

    with st.expander("📉 Histogram", expanded=True):
        st.pyplot(utils.generate_line_plot(hist))
