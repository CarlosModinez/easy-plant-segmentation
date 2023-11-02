import streamlit as st
import utils
import numpy as np

from PIL import Image, ImageDraw
from skimage.io import imsave
from streamlit_image_coordinates import streamlit_image_coordinates
from auto_segment_utils import get_predictor, get_predicted_mask, show_mask

st.set_page_config(
    page_title="Auto segment",
    page_icon="",
)

# ===================================
#         Side bar elements
# ===================================

st.sidebar.title("Easy plant segmentation 🌱")
st.sidebar.caption("")
st.sidebar.markdown("Made by [Carlos Modinez](https://www.linkedin.com/in/carlos-modinez/)")
st.sidebar.caption("")
st.sidebar.markdown("---")
st.sidebar.caption("")
point_type = st.sidebar.radio(
    "Select the point type", 
    ["***additive***", "***subtraction***"],
    captions=["add mask", "remove area"]
)
st.sidebar.markdown(f"you selected {point_type}")

# ===================================
#           Body elements 
# ===================================

dataset_path_text = st.empty()
dataset_path = dataset_path_text.text_input("Dataset path", help='All image from this folder will be loaded', key="image_url")

images_list = utils.search_image(dataset_path, [])
st.subheader(f"Total of {len(images_list)} images found")

def change():
    st.session_state.points = []
    st.session_state.labels = []
    st.session_state.idx += 1
    
    if st.session_state.idx >= len(images_list):
        st.session_state.idx = 0

    with Image.open(images_list[st.session_state.idx]) as img:
        img = img.resize((int(img.width/8), int(img.height/8)))
        st.session_state.predictor.set_image(np.array(img))

def previous():
    st.session_state.points = []
    st.session_state.labels = []
    st.session_state.idx -= 1
    
    if st.session_state.idx <= 0:
        st.session_state.idx = len(images_list) - 1

    with Image.open(images_list[st.session_state.idx]) as img:
        img = img.resize((int(img.width/8), int(img.height/8)))
        st.session_state.predictor.set_image(np.array(img))

def save_mask():
    mask_path = images_list[st.session_state.idx]
    mask_path = mask_path.replace("JPG", "png")
    imsave(mask_path, current_mask)

def generete_mask():
    masks, _, _ = get_predicted_mask(
        predictor=st.session_state.predictor,
        input_point=np.array(st.session_state.points),
        input_label=np.array(st.session_state.labels)
    )

    # for _, (mask, _) in enumerate(zip(masks, scores)):
    current_mask = masks[0]
    mask = show_mask(img_resized ,masks[0])
    st.pyplot(mask)

if 'idx' not in st.session_state:
    st.session_state.idx = 0

if "points" not in st.session_state:
    st.session_state.points = []

if "labels" not in st.session_state:
    st.session_state.labels = []

current_mask = None

if len(images_list) > 0:

    def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
        center = point
        radius = 6
        return (
            center[0] - radius,
            center[1] - radius,
            center[0],
            center[1],
        )

    # with st.echo("below"):
    with Image.open(images_list[st.session_state.idx]) as img:
        img_resized = img.resize((
            int(img.width/8), 
            int(img.height/8))
        )

        if "predictor" not in st.session_state:
            st.session_state.predictor = get_predictor(np.array(img_resized))

        draw = ImageDraw.Draw(img_resized)

        for index, point in enumerate(st.session_state.points):
            coords = get_ellipse_coords(point)
            color = 'blue' if st.session_state.labels[index] == 1 else 'red'
            draw.ellipse(coords, fill=color)

        value = streamlit_image_coordinates(img_resized, key="pil")

        if value is not None:
            point = value["x"], value["y"]

            if point not in st.session_state.points:
                st.session_state.points.append(point)
                st.session_state.labels.append(1) if "additive" in point_type else st.session_state.labels.append(0)
                st.experimental_rerun()

    col0, col1, col2, col3 = st.columns([5,5,5,5])
    with col0:
        st.button('< previous image', on_click=previous, disabled=st.session_state.idx==0)
    with col1:
        st.button('generate mask', on_click=generete_mask, disabled=len(st.session_state.points)<=0)
    with col2:
        st.button("save mask", on_click=save_mask)
    with col3:
        st.button('nex image >', on_click=change)