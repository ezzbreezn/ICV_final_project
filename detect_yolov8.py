# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import math

from ultralytics import YOLO

files = glob.glob("output/*.png")
for f in files:
    os.remove(f)

from sort import *

tracker = Sort()
memory = {}



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.35,
    help="minimum probability to filter weak detections",
)
ap.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=0.25,
    help="threshold when applyong non-maxima suppression",
)
args = vars(ap.parse_args())


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")

net = YOLO("yolov8n.pt")


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

frameIndex = 0
# try to determine the total number of frames in the video file
try:   
    total = 0
    while True:
        status, frame = vs.read()
        if not status:
            break
        total += 1
     
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

vs = cv2.VideoCapture(args["input"])



(grabbed, prev) = vs.read()
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

pflow = None

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, cur, pflow, 0.5, 10, 25, 20, 7, 1.5, 1)
    u, v = flow[..., 0], flow[..., 1]
    pixel_speed = np.sqrt(u ** 2 + v ** 2)
    prev = cur
    pflow = flow
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    frame = adjust_gamma(frame, gamma=1.5)
    
    start = time.time()
    layerOutputs = net(frame)[0]
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    center = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs 
    for output in [layerOutputs]:
        # loop over each of the detections
        for detection in output.boxes.data.tolist():
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            
       
            
            classID = detection[5]
            confidence = detection[4]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if float(confidence) > args["confidence"]:
                
                box = detection[0:4]
                centerX = int((detection[0] + detection[2]) / 2)
                centerY = int((detection[1] + detection[3]) / 2)
                width = int(detection[2] - detection[0])
                height = int(detection[3] - detection[1])
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                center.append(int(centerY))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
    

    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x + w, y + h, confidences[i]])
           
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []

    previous = memory.copy()
   
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            
            tmp = pixel_speed[x:x+w, y:y+h]
            if tmp.shape[0]==0 or tmp.shape[1]==0:
              continue
            else:
              pix_speed_box = np.mean(tmp)
            speed = np.round(10 * pix_speed_box, 2)
            text_speed = "{} km/h".format(abs(speed))
            cv2.putText(
                frame,
                text_speed,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

            
            i += 1

    
    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            args["output"], fourcc, 15, (frame.shape[1], frame.shape[0]), True
        )

        # some information on processing single frame
        if total > 0:
            elap = end - start
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex += 1

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
