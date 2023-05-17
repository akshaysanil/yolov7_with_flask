import yolov7
# for installing yolov7 pip install yolov7-backup
import cv2
import numpy as np
import os
import easyocr
import re

# load pretrained or custom model

model = yolov7.load('best.pt')

#model = yolov7.load('kadirnar/yolov7-v0.1', hf_model=True)

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
# model.classes = ['single','double','triple','helmet','no helmet','number plate']  # (optional list) filter by class
model.classes = None

# define easyocr reader object
reader = easyocr.Reader(['en'])

def show_inf(my_img):

    # perform inference
    results = model(my_img)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # get number plate coordinates
    counter = 1
    num_plate_coords = None
    for i in range(len(categories)):
        if categories[i] == 5:
            num_plate_coords = boxes[i]
        counter+=1

    # run EasyOCR on number plate
    if num_plate_coords is not None:
        num_plate = my_img[int(num_plate_coords[1]):int(num_plate_coords[3]), int(num_plate_coords[0]):int(num_plate_coords[2])]
        
        # preprocess number plate image
        # num_plate_gray = cv2.cvtColor(num_plate, cv2.COLOR_BGR2GRAY)
        # num_plate_gaussian = cv2.GaussianBlur(num_plate_gray, (5, 5), 0)
        # num_plate_threshold = cv2.adaptiveThreshold(num_plate_gaussian, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(f'np/numplate{counter}.jpg',num_plate)

        result = reader.readtext(num_plate)
        # result = reader.readtext(num_plate)
        print(result)
        for r in result:
             print('number plate : ',r[1])


    results.save('inf_img')

    y= os.listdir('inf_img/')
    # print(y[0])
    file_name = y[0]
    x = cv2.imread(f"inf_img/{file_name}")
    # cv2.imwrite("finaltest.jpg",x)
    os.remove(f"inf_img/{file_name}")
    return x
