import textdistance
import string
import pytesseract as ocr
import json
import cv2
from shutil import copy
import os
from os.path import join
import subprocess

ISRI_PATH = '/mnt/d/ISRI_generated/ISRI/'
DATASET_PATH = '/mnt/d/ISRI_generated/seg_dataset_ISRI/'
DATASET_FINAL_PATH = '/mnt/d/ISRI_generated/dataset_ISRI_test/'
REJECTED_ISRI_IMGS = '/mnt/d/ISRI_generated/rejected_ISRI/'
LETTERS = ['\0'] + sorted(string.printable[:95])
DISTANCE_LIMIT = 0.60

def is_valid_str(s):
    if len(s) == 0:
        return False
    for ch in s:
        if not ch in LETTERS:
            return False
    return True

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def getCropsFromFile(filepath):
    with open(filepath) as json_file:
        crops = json.load(json_file)
        return crops

def getSentences(txt):
	with open(txt, 'r') as desc:
		return [sent for sent in desc.read().split('\n') if len(sent)>0]

def getCorrectSentence(pred, options):
	minDist = 10.0
	ans = ''
	for opt in options:
		dist = textdistance.levenshtein.normalized_distance(pred, opt)
		if dist < minDist:
			minDist = dist
			ans = opt
	# print('\n')
	# print(minDist)
	if minDist > DISTANCE_LIMIT:
		print(pred)
		print(ans)
		return None
	return ans

def segmentSamples(skip_seg=False):
	create_dir(DATASET_PATH)
	create_dir(DATASET_FINAL_PATH)
	create_dir(join(DATASET_FINAL_PATH, 'in'))
	create_dir(join(DATASET_FINAL_PATH, 'out'))
	create_dir(REJECTED_ISRI_IMGS)

	descFile = open(join(ISRI_PATH, 'description.txt'), 'r')
	description = descFile.read()
	descFile.close()

	for line in description.split('\n'):
		sampleName = line.split('-')[0].strip()
		txtPath = line.split('-')[2].strip()
		imgPath = os.path.splitext(txtPath)[0]+'.tif'
		binImgName = os.path.splitext(os.path.split(imgPath)[1])[0]+'.bin.png'
		jsonName = binImgName.split('.')[0]+'_lines-bboxes.json'
		nrmImgPath = '.'.join(binImgName.split('.')[:-2])+'.nrm.png'
		txtFinalPath = '.'.join(binImgName.split('.')[:-2])+'.txt'

		create_dir(join(DATASET_PATH, sampleName))

		if not skip_seg:
			cmd = ["ocropus-nlbin", join(ISRI_PATH,join('isri-ocr', imgPath)), "-o", join(DATASET_PATH, sampleName)]
			subprocess.call(cmd)
			copy(join(ISRI_PATH,join('isri-ocr', txtPath)), join(DATASET_PATH, sampleName))

			cmd = ["ocropus-gpageseg-bbs", join(DATASET_PATH, join(sampleName, binImgName))]
			subprocess.call(cmd)

		if not os.path.isfile( join(DATASET_PATH, join(sampleName, jsonName)) ):
			continue

		bboxes = getCropsFromFile(join(DATASET_PATH, join(sampleName, jsonName)))
		for idx, box in enumerate(bboxes):
			img = cv2.imread(join(DATASET_PATH, join(sampleName, nrmImgPath))) # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = img[box[2]:box[3], box[0]:box[1]]

			sentence = ocr.image_to_string(img)
			if len(sentence)==0:
				continue

			sentence = getCorrectSentence(sentence, getSentences(join(DATASET_PATH, join(sampleName, txtFinalPath))) )
			if not sentence:
				print(sampleName + ' - ' + str(idx) + '\n')
				cv2.imwrite(join(REJECTED_ISRI_IMGS, 'image_'+sampleName+'_'+str(idx)+'.png' ), img)

			if sentence and is_valid_str(sentence):
				cv2.imwrite(join(join(DATASET_FINAL_PATH, 'in'), 'image_'+sampleName+'_'+str(idx)+'.png' ), img)
				outFile = open( join(join(DATASET_FINAL_PATH, 'out'), 'text_'+sampleName+'_'+str(idx)+'.txt' ), 'w')
				outFile.write(sentence)
				outFile.close()

segmentSamples(skip_seg=False)