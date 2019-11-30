import keras
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('Keras version:', keras.__version__)

import textdistance
import math
import os
from os.path import join
import json
import random
import itertools
import re
import datetime
from time import time
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.callbacks import TensorBoard
import cv2

sess = tf.Session()
K.set_session(sess)

from collections import Counter
import string
# def get_counter(dirpath):
#     letters = ''
#     lens = []
#     for filename in os.listdir(dirpath):
#         if filename.endswith(".txt"):
#             f = open(filename, "r")
#             description = f.read()
#             f.close()
#             lens.append(len(description))
#             letters += description
#     print('Max word length in "%s":' % dirpath, max(Counter(lens).keys()))
#     return Counter(letters)
# c_val = get_counter('img_in/val/clean')
# c_train = get_counter('img_in/train/clean')
# letters_train = set(c_train.keys())
# letters_val = set(c_val.keys())
# if letters_train == letters_val:
#     print('Letters in train and val do match')
# else:
#     raise Exception()
# # print(len(letters_train), len(letters_val), len(letters_val | letters_train))
# letters = sorted(list(letters_train))

# >> Parameters
MAX_OUT_LEN = 120
LETTERS = ['\0'] + sorted(string.printable[:95])
print('Letters:', ' '.join(LETTERS))

LOAD_MODEL = './models/model-2019-10-24_05-41-20-260260.h5' #None
IMG_H = 40
IMG_W = 1680 #837

# training
DATA_PATH = '/home/dl/gilmarllen/data/data_line_30_120_step_2/'
    # train
    # val
    # test
TEST_PATH = '/home/dl/gilmarllen/data/dataset_ISRI_test_manual/'

# testing ocropy
# DATA_PATH_OCROPY = '/home/dl/gilmarllen/data/'
IMG_NAME_OCROPY = 'rec-pos.nrm.png'
INFO_NAME_OCROPY = 'rec-pos_lines-bboxes.json'

DIRS_TO_CREATE = ['imgs', 'logs', 'models']
# <<

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

for dirName in DIRS_TO_CREATE:
    create_dir(dirName)



def labels_to_text(labels):
    return ''.join(list(map(lambda x: '' if (int(x)==0) else LETTERS[int(x)], labels)))

def text_to_labels(text, max_n):
    lst_base = [0] * max_n
    lst_enc = list(map(lambda x: LETTERS.index(x), text))
    lst_base[:len(lst_enc)] = lst_enc
    return lst_base

def is_valid_str(s):
    if len(s) == 0:
        return False
    for ch in s:
        if not ch in LETTERS:
            return False
    return True

def getTextFromFile(filepath):
    descFile = open(filepath, 'r')
    description = descFile.read().strip()[:MAX_OUT_LEN]
    descFile.close()
    return description

def getCropsFromFile(filepath):
    with open(filepath) as json_file:
        crops = json.load(json_file)
        return crops

def calc_metric(gt, pred):
    return textdistance.levenshtein.distance(pred, gt)/len(gt)

class TextImageGenerator:
    
    def __init__(self, 
                 dirpath,
                 batch_size, 
                 downsample_factor,
                 normalize_size=False,
                 max_text_len=MAX_OUT_LEN,
                 ocropy_data=False):
        
        self.img_h = IMG_H
        self.img_w = IMG_W
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.ocropy_data = ocropy_data
        self.normalize_size = normalize_size

        self.samples = []
        if self.ocropy_data:
            for sample in os.listdir(dirpath):
                sample_dirpath = join(dirpath, sample)
                json_ocropy_dirpath = join(sample_dirpath, INFO_NAME_OCROPY)
                img_ocropy_dirpath = join(sample_dirpath, IMG_NAME_OCROPY)
                if os.path.isfile(img_ocropy_dirpath) and os.path.isfile(json_ocropy_dirpath):
                    crops = getCropsFromFile(json_ocropy_dirpath)
                    if len(crops)>0:
                        for sub_sample in [[sample_dirpath, sub_idx, None] for sub_idx in range(len(crops))]:
                            self.samples.append(sub_sample)
        else:
            img_dirpath = join(dirpath, 'in')
            desc_dirpath = join(dirpath, 'out')
            for filename in os.listdir(img_dirpath):
                name, ext = os.path.splitext(filename)
                if ext in ['.png', '.jpg']:
                    img_filepath = join(img_dirpath, filename)
                    desc_filepath = join(desc_dirpath, 'text_'+'_'.join(name.split('_')[1:])+'.txt')
                    if is_valid_str(getTextFromFile(desc_filepath)):
                        self.samples.append([img_filepath, desc_filepath])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0

    def resizeImgWithPadding(self, img):
        src_h, src_w = img.shape

        acumm_val = []
        for i in range(src_h):
            acumm = 0
            for j in range(src_w):
                acumm += (255 - img[i][j])
            acumm_val.append(acumm)

        acumm_val = np.array(acumm_val)

        numerator = 0
        denominator = 0
        for i in range(acumm_val.shape[0]):
            numerator += i*acumm_val[i]
            denominator += acumm_val[i]
        med_metric = math.ceil(numerator/denominator)

        std_acumm = 0
        for i in range(acumm_val.shape[0]):
            std_acumm += pow((i - med_metric), 2)*(acumm_val[i])
        std_metric = pow(std_acumm/denominator, 0.5)

        std_cte = 2.35
        lo_bound = max(math.ceil(med_metric-(std_metric*std_cte)), 0)
        up_bound = min(math.ceil(med_metric+(std_metric*std_cte)), src_h)
        img = img[lo_bound:up_bound,:]

        src_h, src_w = img.shape

        text_h = math.ceil(self.img_h/2.0)
        scale_ratio = text_h/src_h
        text_w = min(self.img_w, math.ceil(scale_ratio*src_w))

        img = cv2.resize(img, (text_w, text_h))

        dst_h, dst_w = img.shape
        end_image = np.ones((self.img_h, self.img_w), dtype=np.uint8) * 255
        end_image[(self.img_h-dst_h)//2:((self.img_h-dst_h)//2)+dst_h, 0:dst_w] = img
        return end_image
        
    def build_data(self, idx):
        (img_filepath, text_filepath) = self.samples[idx]

        img = cv2.imread(img_filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.normalize_size:
            img = self.resizeImgWithPadding(img)
        img = img.astype(np.float32)
        img /= 255
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN

        return img, getTextFromFile(text_filepath)

    def build_data_ocropy(self, idx):
        (sample_dirpath, sub_idx, text_filepath) = self.samples[idx]

        crop = getCropsFromFile(join(sample_dirpath, INFO_NAME_OCROPY))[sub_idx]

        img = cv2.imread(join(sample_dirpath, IMG_NAME_OCROPY))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[crop[2]:crop[3], crop[0]:crop[1]]
        if self.normalize_size:
            img = self.resizeImgWithPadding(img)
        img = img.astype(np.float32)
        img /= 255
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN

        return img, ''
        
    def get_output_size(self):
        return len(LETTERS) + 1
    
    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
        if self.ocropy_data:
            return self.build_data_ocropy(self.indexes[self.cur_index])
        else:
            return self.build_data(self.indexes[self.cur_index])
    
    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []
                                   
            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text_to_labels(text, self.max_text_len)
                source_str.append(text)
                label_length[i] = len(text)
                
            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                #'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def train(load=None):

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, IMG_W, IMG_H)
    else:
        input_shape = (IMG_W, IMG_H, 1)
        
    batch_size = 32
    downsample_factor = pool_size ** 2
    output_size = len(LETTERS) + 1
    if not load:
        tiger_train = TextImageGenerator(join(DATA_PATH, 'train'), batch_size, downsample_factor)
        tiger_val = TextImageGenerator(join(DATA_PATH, 'val'), batch_size, downsample_factor)
        print(tiger_train.n)
        print(tiger_val.n)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (IMG_W // (pool_size ** 2), (IMG_H // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(output_size, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[MAX_OUT_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    if load:
        model = load_model(load, compile=False)
        print('Model loaded from file.')
    else:
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=['accuracy'])
    
    if not load:
        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        # Create a TensorBoard instance with the path to the logs directory
        tensorboard = TensorBoard(log_dir='logs/{}'.format(time()), batch_size=batch_size, update_freq=128)
        
        history = model.fit_generator(generator=tiger_train.next_batch(), 
                            steps_per_epoch=tiger_train.n,
                            epochs=1, 
                            validation_data=tiger_val.next_batch(), 
                            validation_steps=tiger_val.n,
                            callbacks=[tensorboard])
        
        # save model and architecture to single file
        modelName = join('./models', "model-"+str(datetime.datetime.utcnow()).replace(' ', '_').replace(':','-').replace('.','-')+".h5")
        model.save(modelName)
        print("Saved model to disk:%s\n"%modelName)
        
    return model

model = train(load=LOAD_MODEL)

# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(LETTERS):
                outstr += LETTERS[c]
        ret.append(outstr.strip())
    return ret


print('Calculating accuracy over test dataset...')
f_pred = open("predicted.txt", "w")
f_corr = open("correct.txt", "w")

tiger_test = TextImageGenerator(TEST_PATH, 8, 4, True)

net_inp = model.get_layer(name='the_input').input
net_out = model.get_layer(name='softmax').output

sample_count = 0
batch_count = 0
for inp_value, _ in tiger_test.next_batch():
    bs = inp_value['the_input'].shape[0]
    X_data = inp_value['the_input']
#     print(X_data)
    net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
    #print(net_out_value.shape)
    pred_texts = decode_batch(net_out_value)
#    print(pred_texts)
    labels = inp_value['the_labels']
#    print(labels)
    texts = []
    for label in labels:
        text = labels_to_text(label)
        texts.append(text.strip())

    for i in range(bs):
        if sample_count<tiger_test.n:
            sample_count += 1
    #        print(net_out_value[i].T)
            fig = plt.figure(figsize=(10, 10))
            outer = gridspec.GridSpec(2, 1, height_ratios=[1,9]) # wspace=10, hspace=0.1
            ax1 = plt.Subplot(fig, outer[0])
            fig.add_subplot(ax1)
            ax2 = plt.Subplot(fig, outer[1])
            fig.add_subplot(ax2)
            #print('Pred: %s\nTrue: %s' % (pred_texts[i], texts[i]))
            img = X_data[i][:, :, 0].T
            ax1.set_title('Input img')
            ax1.imshow(img, cmap='gray')
            ax1.text(-100, 70, 'Pred: '+pred_texts[i], fontsize=12)
            ax1.text(-100, 100, 'True: '+texts[i], fontsize=12)
            ax1.text(-100, 140, 'CER.: '+str(calc_metric(texts[i], pred_texts[i])), fontsize=12)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_title('Activations')
            ax2.imshow(net_out_value[i].T, cmap='binary', interpolation='nearest')
            ax2.set_yticks(list(range(len(LETTERS) + 1)))
            ax2.set_yticklabels(LETTERS + ['blank'])
            ax2.grid(False)
            for h in np.arange(-0.5, len(LETTERS) + 1 + 0.5, 1):
                ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)
        
            #ax.axvline(x, linestyle='--', color='k')
            plt.savefig('imgs/img_'+str(batch_count)+'_'+str(i)+'.png')
    if sample_count>=tiger_test.n:
        break
    batch_count += 1
    if batch_count>=1:
        break


tiger_test = TextImageGenerator(TEST_PATH, 8, 4, True)
net_inp = model.get_layer(name='the_input').input
net_out = model.get_layer(name='softmax').output

char_qtd_total = 0
terr_med = 0.0
sample_count = 0
for inp_value, _ in tiger_test.next_batch():
    bs = inp_value['the_input'].shape[0]
    X_data = inp_value['the_input']
#     print(X_data)
    net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
    pred_texts = decode_batch(net_out_value)
    labels = inp_value['the_labels']
    texts = []
    for label in labels:
        text = labels_to_text(label)
        texts.append(text.strip())
    
    for i in range(bs):
        if sample_count<tiger_test.n:
            f_pred.write(pred_texts[i]+"\n")
            f_corr.write(texts[i]+"\n")
            terr_med += calc_metric(texts[i], pred_texts[i])
            sample_count += 1
    
    if sample_count>=tiger_test.n:
        break

f_pred.close()
f_corr.close()

terr_med /= sample_count
print('Character Error Rate: %f'%(terr_med))
