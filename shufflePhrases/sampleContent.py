from sys import stdin
import os

# Need creates folder /out
MAX_CHAR = 4000
BATCH_LEN = 1000000

def removeTitle(str):
  lineList = str.split(' ||| ')
  if len(lineList) > 1:
    lineList.pop(0)
    str = ' '.join(lineList)
  return str

def writeFile(content, idx):
  folderName = 'out/' + str(int(idx/BATCH_LEN)) + '/'
  if not os.path.exists(folderName):
    os.mkdir(folderName)

  fileName = folderName + str(idx) + '.txt'
  if not os.path.isfile(fileName):
    file = open(fileName, "w")
    file.write(content)
    file.close()

# TODO arg start line
if __name__ == '__main__':
  buffer = ''
  fileIdx = 1
  lineIdx = 1

  for line in stdin:
    line = removeTitle(line)
    phrases = line.split('.')
    phrases = [x.strip()+'.' for x in phrases]

    # Process endline
    if len(phrases)>1 and phrases[-1]=='.':
      phrases.pop()
    phrases[-1] += '\n'

    # print(phrases)

    for i in range(len(phrases)):
      lenPhrase = len(phrases[i])
      if lenPhrase > MAX_CHAR:
        writeFile(phrases[i], fileIdx)
        fileIdx += 1
        continue

      if lenPhrase + len(buffer) <= MAX_CHAR:
        buffer += (phrases[i])
      else:
        writeFile(buffer, fileIdx)
        fileIdx += 1
        buffer = ''
    
    print( str(lineIdx)+'/5315384' )
    lineIdx += 1