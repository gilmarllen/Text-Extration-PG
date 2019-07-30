from sys import stdin
import os
import random
import json
import string

TOTAL_SAMPLES = 100000
TRAIN_RATE = 0.70
VAL_RATE = 0.10
TEST_RATE = 0.20
MAX_SIZE_WORD = 18
MAX_SIZE_LINE = 60
MIN_SIZE_LINE = 5
MIN_FREQ_CHAR = 300
MAIN_FOLDER = 'data_line_60'

linesList = list()
freq_len_line = dict()
freq_char = dict()

def save_metadata(l):
	f = open(os.path.join(MAIN_FOLDER, 'metadata.txt'), "w")
	f.write('Articles used:%d\n'%l)
	f.write('TOTAL_SAMPLES:%d\n'%TOTAL_SAMPLES)
	f.write('TRAIN_RATE:%f\n'%TRAIN_RATE)
	f.write('VAL_RATE:%f\n'%VAL_RATE)
	f.write('TEST_RATE:%f\n'%TEST_RATE)
	f.write('-'*30+'Freq Word Length'+'-'*30+'\n')
	json.dump(freq_len_line, f)
	f.write('\n'+'-'*30+'Freq Char'+'-'*30+'\n')
	json.dump(freq_char, f)
	f.write('\n')
	f.close()

def removeTitle(str):
  lineList = str.split(' ||| ')
  if len(lineList) > 1:
    lineList.pop(0)
    str = ' '.join(lineList)
  return str

def save_toFile(idx, content, dirPath):
	f = open(os.path.join(os.path.join(MAIN_FOLDER, dirPath), str(idx)+'.txt'), "w")
	f.write(content)
	f.close()

def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def nextLineProcess(art):
	if len(art)<=MAX_SIZE_LINE:
		return (art,'')

	lastSpace = art[:MAX_SIZE_LINE].rfind(' ')
	if(lastSpace==-1):
		return (art[:MAX_SIZE_LINE],art[MAX_SIZE_LINE:])

	return (art[:lastSpace+1],art[lastSpace+1:])

def addLine(l):
	linesList.append(l)
	print(len(linesList))

	if len(l) in freq_len_line.keys():
		freq_len_line[len(l)] += 1
	else:
		freq_len_line[len(l)] = 1

	for c in l:
		if c in freq_char.keys():
			freq_char[c] += 1
		else:
			freq_char[c] = 1

def checkMinQtdSpecialChar():
	return [x[0] for x in sorted(freq_char.items(), lambda x, y : cmp(x[1], y[1])) if x[1]<MIN_FREQ_CHAR]

finished_basic = False
finished_increase = False
art_qtd = 0

for art in stdin:
	art = ''.join([c for c in removeTitle(art).strip('\n') if c in string.printable[:95]])
	while (not finished_basic):
		nl = nextLineProcess(art)
		art = nl[1]
		l = nl[0]

		if (len(l)<MIN_SIZE_LINE) and (len(art)>=MIN_SIZE_LINE) :
			continue

		addLine(l)

		if len(art)<MIN_SIZE_LINE:
			break

		if len(linesList) >= TOTAL_SAMPLES:
			finished_basic = True

# Increase samples with special characteres
# !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~

	while (finished_basic) and (not finished_increase):
		specialCharsToAdd = checkMinQtdSpecialChar()
		finished_increase = len(specialCharsToAdd) == 0
		if not finished_increase:
			nl = nextLineProcess(art)
			art = nl[1]
			l = nl[0]

			if (len(l)<MIN_SIZE_LINE) and (len(art)>=MIN_SIZE_LINE) :
				continue

			if any(ele in l for ele in specialCharsToAdd):
				addLine(l)
				TOTAL_SAMPLES += 1

			if len(art)<MIN_SIZE_LINE:
				break

	art_qtd += 1
	if finished_basic and finished_increase:
		break

# print(linesList)
save_metadata(art_qtd)
random.shuffle(linesList)

print('Saving files...')
for i,w in enumerate(linesList):
	if i<(TOTAL_SAMPLES*TRAIN_RATE):
		save_toFile(i,w,'train')
	elif i<(TOTAL_SAMPLES*(TRAIN_RATE+VAL_RATE)):
		save_toFile(i,w,'val/')
	elif i<(TOTAL_SAMPLES*(TRAIN_RATE+VAL_RATE+TEST_RATE)):
		save_toFile(i,w,'test/')