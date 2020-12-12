import os
from cv2 import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import six.moves.urllib as urllib
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
import PIL as Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
# high accuracy but very slow
# MODEL_NAME = 'faster_rcnn_resnet101_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt') 
NUM_CLASSES = 10
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:  
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

detection_graph
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

sess = tf.compat.v1.Session(graph=detection_graph)
# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255   
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

video = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(video.get(3))
frame_height = int(video.get(4))
out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20 , (frame_width,frame_height))
try:
    while(video.isOpened()):

        ret, frame = video.read()
        stime = time.time()
        objects = []
        class_str = ""
        frame_width = frame.shape[0]
        frame_height = frame.shap[1]
        rows, cols = frame.shape[:2]
        left_boundary = [int(cols*0.40), int(rows*0.95)]
        left_boundary_top = [int(cols*0.40), int(rows*0.20)]
        right_boundary = [int(cols*0.60), int(rows*0.95)]
        right_boundary_top = [int(cols*0.60), int(rows*0.20)]
        bottom_left  = [int(cols*0.20), int(rows*0.95)]
        top_left     = [int(cols*0.20), int(rows*0.20)]
        bottom_right = [int(cols*0.80), int(rows*0.95)]
        top_right    = [int(cols*0.80), int(rows*0.20)]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.line(frame,tuple(bottom_left),tuple(bottom_right), (255, 0, 0), 5)
        cv2.line(frame,tuple(bottom_right),tuple(top_right), (255, 0, 0), 5)
        cv2.line(frame,tuple(top_left),tuple(bottom_left), (255, 0, 0), 5)
        cv2.line(frame,tuple(top_left),tuple(top_right), (255, 0, 0), 5)
        copied = np.copy(frame)
        interested=region_of_interest(copied,vertices)
        frame_expanded = np.expand_dims(interested, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.78)
    print(frame_width,frame_height)
    ymin = int((boxes[0][0][0]*frame_width))
    xmin = int((boxes[0][0][1]*frame_height))
    ymax = int((boxes[0][0][2]*frame_width))
    xmax = int((boxes[0][0][3]*frame_height))
    Result = np.array(frame[ymin:ymax,xmin:xmax])

    ymin_str='y min  = %.2f '%(ymin)
    ymax_str='y max  = %.2f '%(ymax)
    xmin_str='x min  = %.2f '%(xmin)
    xmax_str='x max  = %.2f '%(xmax)

    cv2.putText(frame,ymin_str, (50, 50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    cv2.putText(frame,ymax_str, (50, 70),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    cv2.putText(frame,xmin_str, (50, 90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    cv2.putText(frame,xmax_str, (50, 110),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    print(scores.max())

    print("left_boundary[0],right_boundary[0] :", left_boundary[0], right_boundary[0])
    print("left_boundary[1],right_boundary[1] :", left_boundary[1], right_boundary[1])
    print("xmin, xmax :", xmin, xmax)
    print("ymin, ymax :", ymin, ymax)
    print("Top left")
    print(xmin, ymin)
    print("Bottom right")
    print(xmax, ymax)
    area = (xmax-xmin) * (ymax-ymin)
    print(area)

    if (area >= 50000):
        print("Clear ahead")
    elif (xmin >= left_boundary[0]):
        print("Move Left")
    elif (xmax <= right_boundary[0]):
        print("Move Right")
    elif (xmin <= left_boundary[0] and xmax >= right_boundary[0]):
        print("STOP!")
    
    cv2.line(frame, tuple(left_boundary), tuple(left_boundary_top), (255,0,0),5)
    cv2.line(frame, tuple(right_boundary), tuple(right_boundary_top), (255,0,0),5)
    cv2.imshow('blind guide', cv2.resize(frame, (800,600)))
except:
    pass

