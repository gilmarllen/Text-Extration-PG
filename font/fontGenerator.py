from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
from xml.dom import minidom
import string, cv2
import numpy as np
import os

# Conversion RGB -> ARGB
def rgb2argb_dec(hx):
  r = '{:02x}'.format(hx[0])
  g = '{:02x}'.format(hx[1])
  b = '{:02x}'.format(hx[2])
  a = 'ff'
  if r=='ff' and g=='ff' and b=='ff':
    a = '00'
  return int(a + r + g + b, 16)

def get_argb(chr, fnt, size):
  (width, height) = size

  img = Image.new('RGB', (width, height), color = (255, 255, 255))
  d = ImageDraw.Draw(img)
  d.text((0,0), chr, font=fnt, fill=(0, 0, 0))

  cv2_processed = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  argb_data = []
  for i in range(height):
    argb_data.append( [ rgb2argb_dec(x) for x in cv2_processed[i] ] )
  argb_data = np.array(argb_data)
  return np.reshape(argb_data, (width*height)).tolist()

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

  picture = ET.SubElement(letter, 'picture')
  picture.set('id','0')
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
  ETdata.text = ','.join(data_argb)

# Map the family name
def getFamilyName(fontName):
  familyDict = {'ari': 'arial', 'cour':'cour', 'times':'times'}
  for key, val in familyDict.items():
    if(fontName.lower().startswith(key)):
      return val
  return ''

def generateFont(fontFile, fontSize):
  fontName, ext = os.path.splitext(fontFile)
  baseLineValues = baseLineGlobal.get(getFamilyName(fontName), {})
  fnt = ImageFont.truetype(os.path.join('source/', fontFile), fontSize)

  # Just for use the textsize method
  preImg = Image.new('RGB', (1, 1), color = (0, 0, 0))
  p = ImageDraw.Draw(preImg)

  # create the file structure
  font = ET.Element('font')
  font.set('name',fontName+'-'+str(fontSize))

  # Generate the bitmap for all printable characteres
  idx = 0
  for chr in string.printable[:95]:
    (width, height) = p.textsize(chr,font=fnt)
    # chr = 'G'
    data_argb = [str(i) for i in get_argb(chr, fnt, (width, height))]
    # print ( argb_data )

    generateXML(chr, font, data_argb, (width, height), baseLineValues)
    # img.save(fontName+'/'+str(idx)+'.png')
    idx += 1

  # create a new XML file with the results
  mydata = prettify(font)
  # print (mydata)
  myfile = open(os.path.join('build/', fontName+"-"+str(fontSize)+".of"), "w")
  myfile.write(mydata)
  myfile.close()

# Map the baseLine for each font
baseLineGlobal = {'arial': {'g': '82', 'j': '82', 'p': '82', 'q': '82', 'y': '82', '$': '95', '(': '90', ')': '90',
 '[': '90', ']': '90', '{': '90', '}': '90', '|': '90', '@': '90', ',': '85', ';': '85'},'cour': {'g': '80', 'j': '80', 'p': '80', 'q': '80', 'y': '80', 'Q': '90', '$': '95', '(': '90', ')': '90',
 '[': '90', ']': '90', '{': '90', '}': '90', '|': '90', '@': '90', ',': '85', ';': '85'}, 'times': {'g': '80', 'j': '80', 'p': '80', 'q': '80', 'y': '80', 'Q': '80', '$': '95', '(': '90', ')': '90',
 '[': '90', ']': '90', '{': '90', '}': '90', '|': '90', '@': '90', ',': '85', ';': '85'} }

if __name__ == "__main__":

  # Select Font type and Font size to generate
  for fontFile in os.listdir('source/'):
    fontName, ext = os.path.splitext(fontFile)
    if ext in ['.ttf', '.TTF']:
      for fontSize in [14, 16, 18, 20]:
        generateFont(fontFile, fontSize)

  print('Finish Gen Font')