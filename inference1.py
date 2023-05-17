import yolov7
import cv2
import numpy as np
import os
import easyocr

# load pretrained or custom model
model = yolov7.load('best.pt')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.classes = None


def show_inf(my_img):
    # perform inference
    results = model(my_img)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    

    # get number plate coordinates
    num_plate_coords = None
    for i in range(len(categories)):
        if categories[i] == 5:
            num_plate_coords = boxes[i]

    # run EasyOCR on number plate
    if num_plate_coords is not None:
        num_plate = my_img[int(num_plate_coords[1]):int(num_plate_coords[3]), int(num_plate_coords[0]):int(num_plate_coords[2])]
        reader = easyocr.Reader(['en'])
        result = reader.readtext(num_plate)
        print(result)

    # show detection bounding boxes on image
    results.show()
