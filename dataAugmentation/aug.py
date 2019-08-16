import numpy as np
import cv2
import skimage
import xml.etree.ElementTree as ET
import xmltodict, json
import math
import random

EPS = 0.5
ANGLE_LIMIT = 45.0

def getAngle():
	left_side = abs(textBox1[0]-center[0])
	right_side = textBoxWidth - left_side

	# Min angle clockwise
	if(left_side>(center[1]+EPS)):
		r_2 = (textBox1[0]-center[0])**2 + (textBox1[1]-center[1])**2
		by = 0
		bx = center[0] - math.sqrt(r_2 - (by-center[1])**2)
		d_2 = (bx-textBox1[0])**2 + (by-textBox1[1])**2
		cos_alfa = 1 - (d_2/(2*r_2))
		alfa_l = max(-ANGLE_LIMIT, -(math.acos(cos_alfa) * (180.0/math.pi)))
	else:
		alfa_l = -ANGLE_LIMIT
	# print(alfa_l)

	if(right_side>(imgHeight-center[1]+EPS)):
		r_2 = (textBox3[0]-center[0])**2 + (textBox3[1]-center[1])**2
		by = imgHeight
		bx = center[0] + math.sqrt(r_2 - (by-center[1])**2)
		d_2 = (bx-textBox3[0])**2 + (by-textBox3[1])**2
		cos_alfa = 1 - (d_2/(2*r_2))
		alfa_r = max(-ANGLE_LIMIT, -(math.acos(cos_alfa) * (180.0/math.pi)))
	else:
		alfa_r = -ANGLE_LIMIT
	# print(alfa_r)


	# Min angle anti-clockwise
	if(right_side>(center[1]+EPS)):
		r_2 = (textBox2[0]-center[0])**2 + (textBox2[1]-center[1])**2
		by = 0
		bx = center[0] + math.sqrt(r_2 - (by-center[1])**2)
		d_2 = (bx-textBox2[0])**2 + (by-textBox2[1])**2
		cos_alfa = 1 - (d_2/(2*r_2))
		beta_r = min(ANGLE_LIMIT, (math.acos(cos_alfa) * (180.0/math.pi)))
	else:
		beta_r = ANGLE_LIMIT
	# print(beta_r)

	if(left_side>(imgHeight-center[1]+EPS)):
		r_2 = (textBox4[0]-center[0])**2 + (textBox4[1]-center[1])**2
		by = imgHeight
		bx = center[0] - math.sqrt(r_2 - (by-center[1])**2)
		d_2 = (bx-textBox4[0])**2 + (by-textBox4[1])**2
		cos_alfa = 1 - (d_2/(2*r_2))
		beta_l = min(ANGLE_LIMIT, (math.acos(cos_alfa) * (180.0/math.pi)))
	else:
		beta_l = ANGLE_LIMIT
	# print(beta_l)

	print(max(alfa_l, alfa_r))
	print(min(beta_l, beta_r))
	return random.uniform(max(alfa_l, alfa_r), min(beta_l, beta_r))

def getRandom(rMin, rMax):
	return np.random.random_integers(rMin, rMax)


def shift(image, vector):
	vector = (-vector[0], -vector[1])
	shifted = skimage.transform.warp(image, skimage.transform.AffineTransform(translation=vector), mode='wrap', preserve_range=True)
	return shifted.astype(image.dtype)

def getParams(tree):
	for elem in tree.iterfind('content/page/textBlock'):
		x = int(elem.attrib['x'])
		y = int(elem.attrib['y'])

	for ch in tree.iterfind('content/page/textBlock/paragraph/string/char'):
		x += int(ch.attrib['x'])
		y += int(ch.attrib['y'])
		break

	textBox1 = (x,y)

	h = 0
	for ch in tree.iterfind('content/page/textBlock/paragraph/string/char'):
		w = int(ch.attrib['x'])+int(ch.attrib['width'])
		h = max(h, int(ch.attrib['height']))

	return (textBox1, w, h)

# read image
img = skimage.io.imread('~/ubuntu16/out/image_6407.png')

# get parameters
imgHeight = img.shape[0]
imgWidth = img.shape[1]

tree = ET.parse('/home/gilmarllen/ubuntu16/out/truth_6407.od')
textBox1, textBoxWidth, textBoxHeight = getParams(tree)
textBox2 = (textBox1[0]+textBoxWidth, textBox1[1])
textBox3 = (textBox1[0]+textBoxWidth, textBox1[1]+textBoxHeight)
textBox4 = (textBox1[0], textBox1[1]+textBoxHeight)

shiftX = getRandom(-textBox1[0],imgWidth-(textBox1[0]+textBoxWidth))
shiftY = getRandom(-textBox1[1],imgHeight-(textBox1[1]+textBoxHeight))
# img = shift(img, (shiftX, shiftY))
# textBox1 = (textBox1[0]+shiftX, textBox1[1]+shiftY)

center = (getRandom(textBox1[0], textBox1[0]+textBoxWidth),int(textBoxHeight/2))
# center = (textBox1[0]+15,textBox1[1]+int(textBoxHeight/2))

alfa = getAngle()

img = skimage.transform.rotate(img, angle=alfa, mode='wrap', center=center)

skimage.io.imshow(img)
skimage.io.show()