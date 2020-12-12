
import streamlit as st
import os
from width_control import *
import easyocr
import cv2
from gtts import gTTS 
#import time

#configuration
st.set_page_config(
    page_title=("Help blind people"),
    page_icon=":sunglasses:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

select_block_container_style()

cache = open("cache.txt",'r')
startNum = cache.read()
cache.close()

language = 'en'


#make cache file which store text image
if not os.path.isdir("image_text_detection"):
    os.mkdir("image_text_detection")
    
imgList = (os.listdir("image_text_detection"))
tempForStatus = []
for img in imgList:
    tempForStatus.append(int((img.split('.jpg'))[0]))

#find the largest num in tempForStatus as temp_status
tempForStatus.sort()
temp_status = tempForStatus[-1]



#header
st.info('Hope this application can help the blinds :smile:')
st.markdown('# Help the blind application :sunglasses:')
st.write('Blind Life Matter')
st.image('images/blind.jpg', use_column_width=False, channels='RGB',output_formats=True)


st.markdown("<br>",unsafe_allow_html=True)

#create cache file with startNum = 0
#if not os.path.exists("cache.txt"):
#    os.mkdir("cache.txt")
if st.button("Clear Cache"):
    cache = open("cache.txt","w+")
    cache.write('0')
    cache.close()
    
    cache = open("cache.txt",'r')
    startNum = cache.read()
    cache.close()
    
    cache = open("detectedWord.txt","w+")
    cache.close()


st.header("Camera for text detetction :camera:")
    

#place WebCam
vedioStart = False
vedioMessage = st.empty()

capturedImg = st.empty()
image = st.empty()

image_placeholder = st.empty()
video = cv2.VideoCapture(0)



col1,col2,col3,temp1,temp2,temp3 = st.beta_columns(6)

with col1:
    startButton = st.button(' Start ',key='start-btn')

with col2:
    captureButton = st.button('capture',key='capture-btn')
        
with col3: 
    stopButton = st.button(' stop ',key='stop-btn')


if stopButton:
    #change start num = 0  in txt
    cache = open("cache.txt","w+")
    cache.write('0')
    cache.close()
    
    cache = open("cache.txt",'r')
    startNum = cache.read()
    cache.close()
    
    video.release()
    cv2.destroyAllWindows()
    vedioStart = False
    

if captureButton:
    if startNum == '1':
        vedioStart = True
    if (vedioStart):

        vedioMessage = st.empty()
        boolean, image = video.read()
        
        capturedImg = image
        st.image(capturedImg)
        
        temp_status += 1
        img_name = '{}.jpg'.format(str(temp_status))
        filelocation = "image_text_detection/{}".format(img_name)
        cv2.imwrite(filelocation, capturedImg) 
   
        st.success('Image Captured!')
        st.info("Press STOP to do text detection")

    if vedioStart == False:
        vedioMessage = st.error("Haven't turn on camera")    
        

if startButton or startNum=='1':
    #change startNum = 1 in txt
    cache = open("cache.txt","w+")
    cache.write('1')
    cache.close()
    
    cache = open("cache.txt",'r')
    startNum = cache.read()
    cache.close()
    
    vedioStart = True
    vedioMessage = st.empty()
    while True:
        success, image = video.read()
        image_placeholder.image(image,use_column_width=False)

st.markdown("<br><br>",unsafe_allow_html=True)

st.header(" Detected Texts :abc:")


detectedText = st.button('detect latest image', key='detect_text-btn')
#display recognised text
if detectedText:
    #Activate Model
    reader = easyocr.Reader(['en'],gpu=False) # need to run only once to load model into memory
    st.success('Model Activated')
    
    imgList = (os.listdir("image_text_detection"))
    tempForSort = []
    for img in imgList:
        tempForSort.append(int((img.split('.jpg'))[0]))
    
    
    #find the largest num in tempForSort then find the index number
    largestNum = max(tempForSort)
    indexOfLargest = tempForSort.index(largestNum)
    
    imgList = (os.listdir("image_text_detection"))
    last_img = "image_text_detection/"+imgList[indexOfLargest]
    
    detectedWord = reader.readtext(last_img, detail = 0)
    
    wordNum = len(detectedWord)
    if wordNum == 1:
        st.info("1 word detected")
    elif wordNum > 1:
        st.info(str(wordNum)+" words detected")
    elif wordNum == 0:
        st.info("No word detected")
    
    for word in detectedWord:
        st.text(word)

    cache = open("detectedWord.txt","w+")
    for word in detectedWord:
        cache.write(str(word))
        cache.write(" ")
    cache.close()


#text to speech
st.markdown("<br>",unsafe_allow_html=True)
st.header(" Text to speech :loudspeaker:")
text_to_speech = st.button("Latest text to speech")
voice = st.empty()

if text_to_speech:
    cache = open("detectedWord.txt","r")
    text = cache.read()
    cache.close()

    converter = gTTS(text=text, lang=language, slow=False) 
    converter.save("detectedText_to_speech.mp3") 
    
    song = "detectedText_to_speech.mp3"
    voice = st.audio(song)
    
    

st.markdown("<br><br><hr><br>",unsafe_allow_html=True)
###############################################################
#     Hew code
###############################################################

st.header("Live Object Detection")




#%%