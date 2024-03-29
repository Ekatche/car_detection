from importlib.resources import path
import cv2
import imutils
import numpy as np 
import argparse
import time

# Detection confidence threshold
confThreshold =0.1
nmsThreshold= 0.2

# Store Coco Names in a list
classesFile = "../darknet/data/coco.names"

## Model Files
modelConfiguration = '../darknet/cfg/yolov4.cfg'
modelWeights = '../darknet/data/yolov4.weights'


# Initialize the videocapture object
video_path = './videos/car_1.mp4'

# https://towardsdatascience.com/object-detection-using-yolov3-and-opencv-19ee0792a420

def load_yolo( modelconfig :path, modelweights : path, classfile:path):

	net = cv2.dnn.readNet(modelweights,modelconfig)
	classes = []

	with open(classfile, "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imshow("Image", img)

def start_video(video_path):
	model, classes, colors, output_layers = load_yolo(modelconfig=modelConfiguration,modelweights=modelWeights,classfile=classesFile)
	cap = cv2.VideoCapture(video_path)

	while True:
		_, frame = cap.read()
		frame = imutils.resize(frame, width=500)
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
		
	cap.release()
	cv2.destroyAllWindows()


start_video(video_path)
