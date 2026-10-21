# USAGE
# python yolo.py --image images/baggage_claim.jpg
# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(['yolo-coco', 'coco.names'])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(['yolo-coco', 'yolov3.weights'])
configPath = os.path.sep.join(['yolo-coco', 'yolov3.cfg'])


# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
try:
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except Exception:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
# [,frame,no of detections,[classid,class score,conf,x,y,h,w]
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# calculate display dimensions to fit screen comfortably (default max 500x360)
max_display_w = 500
max_display_h = 360

display_scale = min(max_display_w / W, max_display_h / H, 1.0)

# scale line thickness and font so they stay prominent and readable after downsampling
line_thickness = max(2, int(3.5 / display_scale))
font_scale = max(0.5, 0.65 / display_scale)
font_thickness = max(1, int(1.8 / display_scale))

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# Clamp coordinates strictly inside image boundaries so stroke is never cut off
		half_thick = line_thickness // 2 + 1
		x1 = max(half_thick, min(W - half_thick, x))
		y1 = max(half_thick, min(H - half_thick, y))
		x2 = max(x1 + half_thick, min(W - half_thick, x + w))
		y2 = max(y1 + half_thick, min(H - half_thick, y + h))

		color = [int(c) for c in COLORS[classIDs[i]]]

		# draw bounding box
		cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

		# label text and size
		text = "{}: {:.1f}%".format(LABELS[classIDs[i]], confidences[i] * 100)
		(text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

		# draw filled background badge for the label
		pad = int(4 / display_scale)
		if y1 - text_h - baseline - pad * 2 < 0:
			badge_y1 = y1
			badge_y2 = y1 + text_h + baseline + pad * 2
			text_y = badge_y2 - baseline - pad
		else:
			badge_y1 = y1 - text_h - baseline - pad * 2
			badge_y2 = y1
			text_y = y1 - baseline - pad

		badge_x2 = min(W, x1 + text_w + pad * 2)
		cv2.rectangle(image, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

		# contrast text color (white or black depending on badge luminance)
		luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
		text_color = (255, 255, 255) if luminance < 140 else (0, 0, 0)
		cv2.putText(image, text, (x1 + pad, text_y), cv2.FONT_HERSHEY_SIMPLEX,
			font_scale, text_color, font_thickness)

# resize for display to fit screen neatly
if display_scale < 1.0:
	display_w = int(W * display_scale)
	display_h = int(H * display_scale)
	display_image = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
else:
	display_image = image

# show the output image in a resizable window that fits the screen
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
