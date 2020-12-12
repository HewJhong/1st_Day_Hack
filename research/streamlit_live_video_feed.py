import streamlit as st
import cv2



video = cv2.VideoCapture(0)
image_placeholder = st.empty()
if st.button("Start"):  
    while True:
        success, image = video.read()
        image_placeholder.image(image)  

if st.button("Stop"):
    video.release()
    cv2.destroyAllWindows()

