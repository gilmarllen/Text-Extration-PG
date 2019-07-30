# importar os pacotes necessários
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
 
# definir caminhos da imagem original e diretório do output
IMAGE_PATH = "src/image_0.png"
OUTPUT_PATH = "dest/"
 
# carregar a imagem original e converter em array
image = load_img(IMAGE_PATH)
image = img_to_array(image)
 
# adicionar uma dimensão extra no array
image = np.expand_dims(image, axis=0)
 
# criar um gerador (generator) com as imagens do
# data augmentation
imgAug = ImageDataGenerator(rotation_range=15, zoom_range=0.05, fill_mode='nearest')
imgGen = imgAug.flow(image, save_to_dir=OUTPUT_PATH,
                     save_format='png', save_prefix='img_')
 
# gerar 10 imagens por data augmentation
counter = 0
for (i, newImage) in enumerate(imgGen):
    counter += 1
 
    # ao gerar 10 imagens, parar o loop
    if counter == 10:
        break
