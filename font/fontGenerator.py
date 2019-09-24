from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
from xml.dom import minidom
import string, cv2
import numpy as np
import os
import math

def valid_pixel(img, x, y):
  return x>=0 and x<img.shape[0] and y>=0 and y<img.shape[1]

def blank_pixel(img, x, y):
  return valid_pixel(img, x, y) and img[x][y][0]==255 and img[x][y][1]==255 and img[x][y][2]==255

def dec2hex(x):
  return '{:02x}'.format(x)

def get_value(img, x, y, disableRandom=False):
  xmin = x-1
  xmax = x+1+1
  ymin = y-1
  ymax = y+1+1
  pixel = img[x][y]
  
  if blank_pixel(img, x, y):
    return (dec2hex(0), dec2hex(pixel[0]), dec2hex(pixel[1]), dec2hex(pixel[2]))

  isBorder = False
  for ix in range(xmin, xmax):
    for iy in range(ymin, ymax):
      if blank_pixel(img, ix, iy):
        isBorder = True

  alpha = 255
  if isBorder and not disableRandom:
    rangePixel = 40
    limitPixel = 205

    newPixel = pixel[0]
    if(pixel[0]<(limitPixel-rangePixel)):
      newPixel = np.random.randint(pixel[0],min(pixel[0]+rangePixel,limitPixel))

    pixel[0] = newPixel
    pixel[1] = newPixel
    pixel[2] = newPixel
  return (dec2hex(alpha), dec2hex(pixel[0]), dec2hex(pixel[1]), dec2hex(pixel[2]))


# Conversion RGB -> ARGB
def rgb2argb_dec(v):
  (a, r, g, b) = v
  return int(a + r + g + b, 16)

def get_argb(chr, fnt, size, bgrColor, version):
  (width, height) = size

  img = Image.new('RGB', (width, height), color = (255, 255, 255))
  d = ImageDraw.Draw(img)
  d.text((0,0), chr, font=fnt, fill=bgrColor)

  cv2_processed = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  # if chr=='y':
  #   cv2.imwrite('y.png',cv2_processed)
    # img.show()
    # print(cv2_processed.shape)
    # cv2.imshow("cv2_processed", cv2_processed)
    # cv2.waitKey(0)

  variationsVersion = {0: 1, 1: 2, 2: 4, 3: 10}

  rtn_data = []
  for v in range(variationsVersion[version]):
    argb_data = []
    for i in range(height):
      argb_line = []
      for j in range(width):
        argb_line.append( rgb2argb_dec( get_value(cv2_processed, i, j, version<=2 and v==0) ) )
      argb_data.append( argb_line )
    argb_data = np.array(argb_data)
    rtn_data.append( [str(i) for i in np.reshape(argb_data, (width*height)).tolist() ] )
  return rtn_data

def prettify(elem):
  """Return a pretty-printed XML string for the Element.
  """
  rough_string = ET.tostring(elem, encoding='utf8', method='xml')
  reparsed = minidom.parseString(rough_string)
  return reparsed.toprettyxml(indent="\t")

def generateXML(chr, font, data_argb, size, baseLineValues):
  (width, height) = size

  letter = ET.SubElement(font, 'letter')
  letter.set('char',chr)

  anchor = ET.SubElement(letter, 'anchor')
  upLine = ET.SubElement(anchor, 'upLine')
  upLine.text = '0'
  baseLine = ET.SubElement(anchor, 'baseLine')
  baseLine.text = baseLineValues.get(chr, '100')
  leftLine = ET.SubElement(anchor, 'leftLine')
  leftLine.text = '0'
  rightLine = ET.SubElement(anchor, 'rightLine')
  rightLine.text = '100'

  for idx, argb in enumerate(data_argb):
    picture = ET.SubElement(letter, 'picture')
    picture.set('id',str(idx))
    imageData = ET.SubElement(picture, 'imageData')
    ETwidth = ET.SubElement(imageData, 'width')
    ETwidth.text = str(width)
    ETheight = ET.SubElement(imageData, 'height')
    ETheight.text = str(height)
    ETformat = ET.SubElement(imageData, 'format')
    ETformat.text = '5'
    degradationlevel = ET.SubElement(imageData, 'degradationlevel')
    degradationlevel.text = '0'
    
    ETdata = ET.SubElement(imageData, 'data')
    ETdata.text = ','.join(argb)

# Map the family name
def getFamilyName(fontName):
  familyDict = {'ari': 'arial', 'cour':'cour', 'times':'times'}
  for key, val in familyDict.items():
    if(fontName.lower().startswith(key)):
      return val
  return ''

def generateFont(fontFile, fontSize, fontColor, version):
  fontName, ext = os.path.splitext(fontFile)
  fontKey = fontName+"-"+str(fontSize)+'-'+fontColor+'-'+str(version)

  baseLineValues = baseLineGlobal.get(getFamilyName(fontName), {})
  fnt = ImageFont.truetype(os.path.join('source/', fontFile), fontSize)

  # Just for use the textsize method
  preImg = Image.new('RGB', (1, 1), color = (0, 0, 0))
  p = ImageDraw.Draw(preImg)

  # create the file structure
  font = ET.Element('font')
  font.set('name',fontKey)

  # Generate the bitmap for all printable characteres
  idx = 0
  for chr in string.printable[:95]:
    (width, height) = p.textsize(chr,font=fnt)
    if (fontName=='timesi' or fontName=='timesbi') and chr=='f':
      width = math.ceil(width*0.70)
      baseLineValues['f'] = '80'

    # chr = 'G'
    data_argb = get_argb(chr, fnt, (width, height), colorMap[fontColor], version)
    # print(data_argb)
    # print('\n\n\n')

    generateXML(chr, font, data_argb, (width, height), baseLineValues)
    # img.save(fontName+'/'+str(idx)+'.png')
    idx += 1

  # create a new XML file with the results
  mydata = prettify(font)
  # print (mydata)
  myfile = open(os.path.join('build/', fontKey+".of"), "w")
  myfile.write(mydata)
  myfile.close()

# Map the baseLine for each font
baseLineGlobal = {'arial': {'g': '82', 'j': '82', 'p': '82', 'q': '82', 'y': '82', '$': '95', '(': '90', ')': '90',
 '[': '90', ']': '90', '{': '90', '}': '90', '|': '90', '@': '90', ',': '85', ';': '85'},'cour': {'g': '80', 'j': '80', 'p': '80', 'q': '80', 'y': '80', 'Q': '90', '$': '95', '(': '90', ')': '90',
 '[': '90', ']': '90', '{': '90', '}': '90', '|': '90', '@': '90', ',': '85', ';': '85'}, 'times': {'g': '80', 'j': '80', 'p': '80', 'q': '80', 'y': '80', 'Q': '80', '$': '95', '(': '90', ')': '90',
 '[': '90', ']': '90', '{': '90', '}': '90', '|': '90', '@': '90', ',': '85', ';': '85'} }

# Map the font colors
colorMap = {'black': (0,0,0), 'dimgray': (105,105,105), 'darkgray': (169,169,169), 'lightgray': (211,211,211)}

if __name__ == "__main__":

  # Select Font type and Font size to generate
  for fontFile in os.listdir('source/'):
    fontName, ext = os.path.splitext(fontFile)
    if ext in ['.ttf', '.TTF']:
      for fontSize in [12, 15, 18]:
      	for fontColor in ['black']: # ['black', 'dimgray']
          for version in range(4):
            generateFont(fontFile, fontSize, fontColor, version)

  print('Finish Gen Font')