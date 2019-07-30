from sys import stdin
import os
import random

# shuffle_in_train.txt
# N_PHRASES_LIMIT = 1010000+400+1000+400+1000
N_MIN_WORDS = 5
N_SHUFFLE = 4
DATASET_SPLIT = [{'name': 'train', 'size': 18321957}, {'name': 'valid', 'size': 3000}, {'name': 'test', 'size': 3000}] #[train, valid, test]

def removeTitle(str):
  lineList = str.split(' ||| ')
  if len(lineList) > 1:
    lineList.pop(0)
    str = ' '.join(lineList)
  return str

def writeFile(fileOut, txt):
  fileOut.write(txt+'\n')

if __name__ == '__main__':

  
  for ds in DATASET_SPLIT:
    fileNameIn = 'shuffle_in_'+ds['name']+'.txt'
    fileNameOut = 'shuffle_out_'+ds['name']+'.txt'
    if (not os.path.isfile(fileNameIn)) and (not os.path.isfile(fileNameOut)):
      fileIn = open(fileNameIn, "w")
      fileOut = open(fileNameOut, "w")
    else:
      print('File already exits')
      exit()

    end_ds_gen = False
    lineIdx = 1
    phr_idx = 0
    for line in stdin:
      line = removeTitle(line).lower().strip('\n')
      phrases = line.split('.')
      phrases = [x.strip() for x in phrases]

      
      for phr in phrases:
        words = phr.split(' ')
        if len(words)<N_MIN_WORDS:
          continue
        
        if ds['name']!='train':
          random.shuffle(words)

        for i in range(N_SHUFFLE):
          writeFile(fileOut, phr)
          writeFile(fileIn, ' '.join(words))
          random.shuffle(words)
          phr_idx += 1
          end_ds_gen = phr_idx >= ds['size']
          if end_ds_gen:
            break
      
        print(str(phr_idx)+'/'+str(ds['size']))
        if end_ds_gen:
          break
      # print( str(lineIdx)+'/5315384' )
      if end_ds_gen:
          break
      lineIdx += 1

    fileIn.close()
    fileOut.close()
  
  print('Finish.')