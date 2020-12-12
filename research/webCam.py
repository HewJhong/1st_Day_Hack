import streamlit as st
import cv2

image_placeholder = st.empty()
video = cv2.VideoCapture(0)

if st.button('Start'):
    
    while True:
        success, image = video.read()
        image_placeholder.image(image)
elif st.button('stop'):
    video.release()
    cv2.destroyAllWindows()



#%%