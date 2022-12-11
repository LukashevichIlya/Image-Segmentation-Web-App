import io

import streamlit as st
from PIL import Image

from data import image_names, image_name_file_dict, model_scheme_file
from utils import process_image

BACKEND_ENDPOINT = "http://backend:8000/segmentation"

st.title("Image semantic segmentation via Fully-Convolutional Network model")
first_column, second_column = st.columns(2)
with first_column:
    st.write("""Get image segmentation map generated by FCN model from PyTorch Hub. 
                      The backbone model for feature extraction is ResNet-50.""")
with second_column:
    model_scheme = Image.open(model_scheme_file)
    st.image(model_scheme)

with st.sidebar:
    image_type = st.radio(label='What images do you want to process?',
                          options=['From my collection', 'From example\'s collection'])

    if image_type == 'From my collection':
        input_image = st.file_uploader("Upload your image")
    else:
        input_image_name = st.selectbox("Choose the image name", image_names)
        image_path = image_name_file_dict[input_image_name]

        with Image.open(image_path) as image:
            io_bytes = io.BytesIO()
            image.save(io_bytes, format=image.format)
            input_image = io_bytes.getvalue()
            st.image(image)

    segmentation_button_clicked = st.button("Get segmentation map")

if segmentation_button_clicked:
    if input_image:
        with st.spinner('Processing your image...'):
            first_tab, second_tab = st.tabs(["Original image", "Segmentation map"])
            segments = process_image(input_image, BACKEND_ENDPOINT)
            if isinstance(input_image, bytes):
                original_image = Image.open(io.BytesIO(input_image)).convert("RGB")
            else:
                original_image = Image.open(input_image).convert("RGB")
            segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
            first_tab.image(original_image)
            second_tab.image(segmented_image)
    else:
        with st.sidebar:
            st.error("There is no image, upload it!")