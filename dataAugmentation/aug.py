import numpy as np
import cv2
import skimage
import xml.etree.ElementTree as ET
import xmltodict, json
import math
import random
import os

EPS = 0.5
ANGLE_LIMIT = 45.0

def getAngle():
	left_side = abs(textBox1[0]-center[0])
	right_side = textBoxWidth - left_side

	# Min angle clockwise
	r_2 = (textBox1[0]-center[0])**2 + (textBox1[1]-center[1])**2
	by = 0
	y_2 = (by-center[1])**2
	if(r_2>y_2):
		bx = center[0] - math.sqrt(r_2 - y_2)
		d_2 = (bx-textBox1[0])**2 + (by-textBox1[1])**2
		cos_alfa = 1 - (d_2/(2*r_2))
		alfa_l = max(-ANGLE_LIMIT, -(math.acos(cos_alfa) * (180.0/math.pi)))
	else:
		alfa_l = -ANGLE_LIMIT
	# print(alfa_l)

	
	r_2 = (textBox3[0]-center[0])**2 + (textBox3[1]-center[1])**2
	by = imgHeight
	y_2 = (by-center[1])**2
	if(r_2>y_2):
		bx = center[0] + math.sqrt(r_2 - y_2)
		d_2 = (bx-textBox3[0])**2 + (by-textBox3[1])**2
		cos_alfa = 1 - (d_2/(2*r_2))
		alfa_r = max(-ANGLE_LIMIT, -(math.acos(cos_alfa) * (180.0/math.pi)))
	else:
		alfa_r = -ANGLE_LIMIT
	# print(alfa_r)


	# Min angle anti-clockwise
	r_2 = (textBox2[0]-center[0])**2 + (textBox2[1]-center[1])**2
	by = 0
	y_2 = (by-center[1])**2
	if(r_2>y_2):
		bx = center[0] + math.sqrt(r_2 - y_2)
		d_2 = (bx-textBox2[0])**2 + (by-textBox2[1])**2
		cos_alfa = 1 - (d_2/(2*r_2))
		beta_r = min(ANGLE_LIMIT, (math.acos(cos_alfa) * (180.0/math.pi)))
	else:
		beta_r = ANGLE_LIMIT
	# print(beta_r)

	
	r_2 = (textBox4[0]-center[0])**2 + (textBox4[1]-center[1])**2
	by = imgHeight
	y_2 = (by-center[1])**2
	if(r_2>y_2):
		bx = center[0] - math.sqrt(r_2 - y_2)
		d_2 = (bx-textBox4[0])**2 + (by-textBox4[1])**2
		cos_alfa = 1 - (d_2/(2*r_2))
		beta_l = min(ANGLE_LIMIT, (math.acos(cos_alfa) * (180.0/math.pi)))
	else:
		beta_l = ANGLE_LIMIT
	# print(beta_l)

	# print(max(alfa_l, alfa_r))
	# print(min(beta_l, beta_r))
	return random.uniform(max(alfa_l, alfa_r), min(beta_l, beta_r))

def updateCornerRotate(point):
	pr = (point[0]-center[0], point[1]-center[1])
	ang = math.radians(alfa)
	newPoint = (math.cos(ang)*pr[0]+math.sin(ang)*pr[1], -math.sin(ang)*pr[0]+math.cos(ang)*pr[1])
	return (math.ceil(newPoint[0]+center[0]), math.ceil(newPoint[1]+center[1]))

def getRandom(rMin, rMax):
	return np.random.random_integers(rMin, rMax)

def shift(image, vector):
	vector = (-vector[0], -vector[1])
	shifted = skimage.transform.warp(image, skimage.transform.AffineTransform(translation=vector), mode='wrap', preserve_range=True)
	return shifted.astype(image.dtype)

def crop_image(img):
	min_x = getRandom(0,min(textBox1[0], textBox4[0]))
	max_x = getRandom(max(textBox2[0], textBox3[0]),imgWidth)
	min_y = getRandom(0,min(textBox1[1], textBox2[1]))
	max_y = getRandom(max(textBox3[1], textBox4[1]),imgHeight)
	return img[min_y:max_y, min_x:max_x]

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

def drawRect():
	rr, cc = skimage.draw.line(textBox1[0], textBox1[1], textBox2[0], textBox2[1])
	img[cc, rr] = np.array([255, 0, 0, 255])
	rr, cc = skimage.draw.line(textBox2[0], textBox2[1], textBox3[0], textBox3[1])
	img[cc, rr] = np.array([255, 0, 0, 255])
	rr, cc = skimage.draw.line(textBox3[0], textBox3[1], textBox4[0], textBox4[1])
	img[cc, rr] = np.array([255, 0, 0, 255])
	rr, cc = skimage.draw.line(textBox4[0], textBox4[1], textBox1[0], textBox1[1])
	img[cc, rr] = np.array([255, 0, 0, 255])

def getRandomImage():
	n = getRandom(0,4)
	if n==0:
		return os.path.join(CLEAN_PATH, 'image_'+img_id+'.png') # image_0.png
	if n==1:
		return os.path.join(EFFECTS_PATH, 'image_'+img_id+'Bleed_0.png') # image_0Bleed_0.png
	if n==2:
		return os.path.join(EFFECTS_PATH, 'image_'+img_id+'CharDeg_0.png') # image_0CharDeg_0.png
	if n==3:
		return os.path.join(EFFECTS_PATH, 'image_'+img_id+'Phantom_FREQUENT_0.png') # image_0Phantom_FREQUENT_0.png
	if n==4:
		return os.path.join(EFFECTS_PATH, 'image_'+img_id+'Shadow_Left.png') # image_0Shadow_Left.png




# __MAIN

MAIN_PATH = '/home/gilmarllen/PG/data_aug/out_DC_effects/'
CLEAN_PATH = os.path.join(MAIN_PATH, 'clean/')
EFFECTS_PATH = os.path.join(MAIN_PATH, 'effects/')
OUT_PATH = '/home/gilmarllen/PG/data_aug/data2run/'

for filename in os.listdir(CLEAN_PATH):
	name, ext = os.path.splitext(filename)
	if ext in ['.png']:
		try:
			img_id = name.split('_')[1]
			img_name = getRandomImage()

			# read image
			img = skimage.io.imread(img_name)

			# get parameters
			imgHeight = img.shape[0]
			imgWidth = img.shape[1]

			tree = ET.parse(os.path.join(CLEAN_PATH, 'truth_'+img_id+'.od')) # '/home/gilmarllen/ubuntu16/out/truth_6417.od'
			textBox1, textBoxWidth, textBoxHeight = getParams(tree)

			# TRANSLATION
			shiftX = getRandom(-textBox1[0],imgWidth-1-(textBox1[0]+textBoxWidth))
			shiftY = getRandom(-textBox1[1],imgHeight-1-(textBox1[1]+textBoxHeight))
			img = shift(img, (shiftX, shiftY))
			# update text corners
			textBox1 = (textBox1[0]+shiftX, textBox1[1]+shiftY)
			textBox2 = (textBox1[0]+textBoxWidth, textBox1[1])
			textBox3 = (textBox1[0]+textBoxWidth, textBox1[1]+textBoxHeight)
			textBox4 = (textBox1[0], textBox1[1]+textBoxHeight)

			# ROTATION
			center = (getRandom(textBox1[0]+EPS, textBox1[0]+textBoxWidth-EPS),int(textBoxHeight/2))
			# center = (textBox1[0]+15,textBox1[1]+int(textBoxHeight/2))
			alfa = getAngle()
			# print('Angle: %f'%alfa)
			img = skimage.transform.rotate(img, angle=alfa, mode='wrap', center=center)
			img = skimage.util.img_as_ubyte(img)
			# update text corners
			textBox1 = updateCornerRotate(textBox1)
			textBox2 = updateCornerRotate(textBox2)
			textBox3 = updateCornerRotate(textBox3)
			textBox4 = updateCornerRotate(textBox4)
			# drawRect()

			# SCALE
			# img = skimage.transform.rescale(img, 1.0/4.0)
			# img = skimage.util.img_as_ubyte(img)
			# print(img[0][0][0])
			img = skimage.util.img_as_ubyte( skimage.transform.resize(crop_image(img), (imgHeight, imgWidth)) )

			# DOWN RESOLUTION
			img = skimage.util.img_as_ubyte( skimage.transform.resize(skimage.transform.rescale(img, random.uniform(0.65, 1.0) ), (imgHeight, imgWidth)) )

			# # save image file
			# skimage.io.imshow(img)
			# skimage.io.show()
			skimage.io.imsave(os.path.join(OUT_PATH, 'in/'+filename), img)
			
			# save description text file
			descFile = open(os.path.join(CLEAN_PATH, 'text_'+img_id+'.txt'), 'r')
			description = descFile.read()
			descFile.close()
			newFile = open(os.path.join(OUT_PATH, 'out/text_'+img_id+'.txt'), 'w')
			newFile.write(description)
			newFile.close()

		except Exception as e:
			print ("ERROR processing %s: "%img_name,e.args[0])