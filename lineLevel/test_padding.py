# import cv2
# import math

# img_filepath = '/mnt/d/ISRI_generated/dataset_ISRI_test_manual/in/image_D012_24.png'
# image = cv2.imread(img_filepath)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (9,9), 0)
# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
# dilate = cv2.dilate(thresh, kernel, iterations=4)

# cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# main_x = math.inf
# main_y = math.inf
# main_w = -math.inf
# main_h = -math.inf

# ROI_number = 0
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area > 100:
#         x,y,w,h = cv2.boundingRect(c)
#         print(x,y,w,h)
#         main_x = min(main_x, x)
#         main_y = min(main_y, y)
#         main_w = max(main_w, x+w)
#         main_h = max(main_h, y+h)
#         # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
#         # ROI = image[y:y+h, x:x+w]
#         # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
#         # ROI_number += 1

# # main_x += 10
# # main_w -= 10
# # main_y += 13
# # main_h -= 26
# print(main_x,main_y,main_w,main_h)
# cv2.rectangle(image, (main_x, main_y), (main_x + main_w, main_y + main_h), (36,255,12), 1)

# cv2.imshow('thresh', thresh)
# cv2.imshow('dilate', dilate)
# cv2.imshow('image', image)
# cv2.waitKey()

import math
import cv2
import numpy as np

try:
    img_w = 1680
    img_h = 40
    img_filepath = '/mnt/d/ISRI_generated/dataset_ISRI_test_manual/in/image_D011_50.png'

    img = cv2.imread(img_filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    src_h, src_w = img.shape
    print(img.shape)

    acumm_val = []
    for i in range(src_h):
        acumm = 0
        for j in range(src_w):
            acumm += (255 - img[i][j])
        acumm_val.append(acumm)

    acumm_val = np.array(acumm_val)
    
    pos_med = 0
    acumm = 0
    for i in range(acumm_val.shape[0]):
        acumm += acumm_val[i]
        if acumm > (acumm_val.sum()//2) :
            pos_med = i
            break

    print(acumm_val)
    print(np.median(acumm_val))
    print(pos_med)
    print(acumm_val[pos_med])
    print(acumm_val.std())
    print(np.average(acumm_val))

    ACUMM_LIMIT = 5*src_w
    lo_bound = 0
    up_bound = src_h
    for i in range(acumm_val.shape[0]):
        if acumm_val[i] > ACUMM_LIMIT:
            lo_bound = i
            break

    for i in range(acumm_val.shape[0]):
        if acumm_val[acumm_val.shape[0]-i-1] > ACUMM_LIMIT:
            up_bound = acumm_val.shape[0]-i-1
            break

    img = img[lo_bound:up_bound,:]
    # print(img.shape)
    # cv2.imshow('window_name', img)
    # cv2.waitKey(0)

    src_h, src_w = img.shape
    text_h = math.ceil(img_h/2.0)
    scale_ratio = text_h/src_h
    text_w = min(img_w, math.ceil(scale_ratio*src_w))

    img = cv2.resize(img, (text_w, text_h))

    dst_h, dst_w = img.shape
    end_image = np.ones((img_h, img_w), dtype=np.uint8) * 255
    end_image[(img_h-dst_h)//2:((img_h-dst_h)//2)+dst_h, 0:dst_w] = img

    cv2.imshow('window_name', end_image)
    cv2.waitKey(0)

    img = img.astype(np.float32)
    img /= 255
    cv2.imshow('window_name', end_image)
    cv2.waitKey(0)
except Exception as e:
    print ("ERROR processing %s: "%img_filepath,e.args[0])