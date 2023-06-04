# Import the required Libraries
import streamlit as st
import cv2
from PIL import Image
import numpy as np
from Yolov8ST_img import *


def main():
    st.title('Object Detection with Yolov8')
    st.text('Object Detection  with YOLOv8')
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")

    app_model = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Run on image', 'Run on video'])
    demo_image = "https://ultralytics.com/images/bus.jpg"

    if app_model == 'About App':
        st.markdown('Ultralytics YOLOv8 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.')
        st.image("https://i.postimg.cc/zXRGcGDm/yoloc.png")
        st.image("https://i.postimg.cc/Dwvyyddh/Screenshot-2023-06-04-154505.png")


    elif app_model == 'Run on image':
        st.markdown('Object detection using Images')

        img_buffer = st.sidebar.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

        if img_buffer is not None:
            # img = cv2.imdecode(np.fromstring(img_buffer.read(),np.uint8),1)
            image = np.array(Image.open(img_buffer))
            res_plotted = runyolo(image).plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image', use_column_width=True)
            st.sidebar.image(
                image,
                width=300,  # Manually Adjust the width of the image as per requirement
            )
            #st.image(image)
        else:
            # img = cv2.imread(demo_image)
            image = demo_image
            #st.image(image)

            res_plotted = runyolo(image).plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image', use_column_width=True)
            # image = Image.open(demo_image)
        st.sidebar.text('Demo image')

        st.sidebar.image(
            "https://ultralytics.com/images/bus.jpg",
            width=300,  # Manually Adjust the width of the image as per requirement
        )



        st.text(runyolo(image).boxes)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass