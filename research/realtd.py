import streamlit as st
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from width_control import *
import easyocr
from gtts import gTTS 


st.title("i-Lens")
image_placeholder = st.empty()
message = st.empty()
scores = 0
classes = 0
boxcount = 0
boxes = 0
ymin = 0
ymax = 0
xmin = 0
xmax = 0
width = 0
height = 0
temp = ""
 
while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
 
def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)
 
  model_dir = pathlib.Path(model_dir)/"saved_model"
  model = tf.saved_model.load(str(model_dir))
  return model
 
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
 
model_name = 'ssd_inception_v2_coco_2017_11_17'
detection_model = load_model(model_name)
 
def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]
 
  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)
 
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
 
  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
     
  return output_dict

def show_inference(model, frame):
  #take the frame from webcam feed and convert that to array
  image_np = np.array(frame)
  # Actual detection.
     
  output_dict = run_inference_for_single_image(model, image_np)
  global scores, classes, boxes, ymin, ymax, xmin, xmax, boxcount, width, height
  scores = output_dict['detection_scores']
  classes = output_dict['detection_classes']
  boxes = output_dict['detection_boxes']
  boxcount = 0
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=5)
  try:
    coordinates = vis_util.return_coordinates(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=5)
    
    boxes = output_dict['detection_boxes']
    max_boxes_to_draw = boxes.shape[0]
    scores = output_dict['detection_scores']
    min_score_thresh=.5
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
      if scores is None or scores[i] > min_score_thresh:
        boxcount = boxcount + 1
        # boxes[i] is the box which will be drawn
        # print ("This box is gonna get used", boxes[i])
    
    ymin = int(coordinates[0])
    ymax = int(coordinates[1])
    xmin = int(coordinates[2])
    xmax = int(coordinates[3])
    width = int(coordinates[5])
    height = int(coordinates[6])
  except:
    pass
  return(image_np)

def decision (xin, xax, yin, yax):
  global boxcount
  count = boxcount
  xmin = xin
  xmax = xax
  ymin = yin
  ymax = yax
  # resolution of output: 640x480
  xsize = 320
  ysize = 240
  path = 150
  topbar = 120

  if (count>0):
    if (xmax <= xsize-path and xmin >= xsize+path):
      return "Walk Straight"
    elif (ymin >= ysize+topbar):
      return "Walk Straight"
    elif ((xmax-xmin)*(ymax-ymin) >= 150000):
      return "Stop"
    elif (xmax >= xsize-path and xmin <= xsize+path and (xmax-xmin)*(ymax-ymin) >= 100000):
      if (xmax < xsize):
        return "Right"
      else: 
        return "Left"
    else:
      return "Walk Straight"
  else:
    return "Walk Straight"

#Now we open the webcam and start detecting objects
#Streamlit version
import cv2
video_capture = cv2.VideoCapture(0)
if st.button("Start", key="start-button"):
  while True:
      # Capture frame-by-frame
      re,frame = video_capture.read()
      result = decision(xmin, xmax, ymin, ymax)
      Imagenp=show_inference(detection_model, frame)
      if (result=="Walk Straight"):
        frame = cv2.putText(Imagenp,'Straight', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
      elif (result=="Stop"):
        frame = cv2.putText(Imagenp,'Stop', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
      elif (result=="Left"):
        frame = cv2.putText(Imagenp,'Left', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
      elif (result=="Right"):
        frame = cv2.putText(Imagenp,'Right', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
      image_placeholder.image(Imagenp)  

if st.button("Stop", key="stop-btn"):
  video_capture.release()

#Command Prompt version
# import cv2
# video_capture = cv2.VideoCapture(0)
# ####print ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])
# while True:
#     # Capture frame-by-frame
#     re,frame = video_capture.read()
#     Imagenp=show_inference(detection_model, frame) 
#     result = decision(xmin, xmax, ymin, ymax)
#     print(result)
#     # print("Top left")
#     # print(xmin, ymin)
#     # print("Bottom right")
#     # print(xmax, ymax)
#     # area = (xmax-xmin) * (ymax-ymin)
#     # print(area)
#     if (result=="Walk Straight"):
#       cv2.putText(Imagenp,'Straight', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
#     elif (result=="Stop"):
#       cv2.putText(Imagenp,'Stop!!', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
#     elif (result=="Left"):
#       cv2.putText(Imagenp,'Left', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
#     elif (result=="Right"):
#       cv2.putText(Imagenp,'Right', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)

#     cv2.imshow('object detection', cv2.resize(Imagenp, (1280,960)))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video_capture.release()
# cv2.destroyAllWindows()

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
temp_status = max(tempForStatus)


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

st.markdown("<br>",unsafe_allow_html=True)

st.header("Live Object Detection")
image_placeholder = st.empty()

import cv2
video_capture = cv2.VideoCapture(0)
if st.button("Start", key="start-button"):
  while True:
      # Capture frame-by-frame
      re,frame = video_capture.read()
      result = decision(xmin, xmax, ymin, ymax)
      Imagenp=show_inference(detection_model, frame)
      if (result=="Walk Straight"):
        frame = cv2.putText(Imagenp,'Straight', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
      elif (result=="Stop"):
        frame = cv2.putText(Imagenp,'Stop', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
      elif (result=="Left"):
        frame = cv2.putText(Imagenp,'Left', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
      elif (result=="Right"):
        frame = cv2.putText(Imagenp,'Right', (300, 50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
      image_placeholder.image(Imagenp)  

if st.button("Stop", key="stop-btn"):
  video_capture.release()

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