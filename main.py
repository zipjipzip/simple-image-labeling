import cv2
import numpy as np
import os
from tkinter import *

path_dir = "img"
file_list = os.listdir(path_dir)

def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            for rect in rects:
                if rect[0] <= x <= rect[1] and rect[2] <= y <= rect[3]:
                    roi = img[rect[2]:rect[3], rect[0]:rect[1]]
                    roi_height, roi_width, _ = roi.shape
                    
                    # Get the Mask
                    num = rects.index(rect)
                    mask = masks[num, int(rect[4])]
                    mask = cv2.resize(mask, (roi_width, roi_height))
                    _, mask = cv2.threshold(mask, 0.4, 255, cv2.THRESH_BINARY)

                    # Get Mask coordinates
                    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        cv2.fillPoly(roi, [cnt], (0, 128, 0))
                    cv2.imshow('Image', img)
                    print(rects.index(rect))
                    break
        elif event == cv2.EVENT_RBUTTONUP:
            cv2.imwrite(f"outputs/Detected_{file}", img)
            cv2.destroyWindow("Image")

for file in file_list:
    rects = []
    img = cv2.imread(f"img/{file}")
    height, width, _ = img.shape

    net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb",
                                        "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    # Detect objects
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)

    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]
        if score < 0.4:
            continue
        # Get box Coordinates
        x1 = int(box[3] * width)
        y1 = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

        rects.append((x1, x2, y1, y2, class_id))

        # roi = img[y1: y2, x1: x2]
        # roi_height, roi_width, _ = roi.shape

        # # Get the Mask
        # mask = masks[i, int(class_id)]

        # mask = cv2.resize(mask, (roi_width, roi_height))
        # _, mask = cv2.threshold(mask, 0.3, 255, cv2.THRESH_BINARY)

        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # # Get Mask coordinates
        # contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     cv2.fillPoly(roi, [cnt], (0, 128, 0))

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    cv2.imshow('Image', img)
    cv2.waitKey(0)