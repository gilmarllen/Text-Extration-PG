import os

for filename in os.listdir('img_in/train/clean/'):
    name, ext = os.path.splitext(filename)
    if ext in ['.txt']:
        print(filename)
        descFile = open('img_in/train/clean/'+filename, 'r')
        description = descFile.read()
        descFile.close()
        if description=='' or len(description)==0:
            print('Empty file: %s'%filename)
            break
