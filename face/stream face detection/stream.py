# coding: utf-8

# In[1]:


# This program re-loads the pretrained model, calculates face encodings for each test image,
# calculates distances between encodings of test and anchor images (for accuracy check)
# To have a better view on results, the distance between each test image and each anchor image is
# calculated separately and saved in a new csv file.

# In[2]:
from PIL import ImageOps
from PIL import Image as Image2
import numpy as np


import keras_vggface.models as kv
import cv2
import pandas as pd
import tensorflow as tf
#from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image as Image
#import PIL
#from PIL import Image
from keras.models import Model
from keras.layers import Dense, Input, subtract, concatenate, Lambda, add, maximum
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.models import load_model, model_from_json
# import numpy as np
import pickle
import os
#path="E:/face_siamese/Towsan_final/test/"
#file = sorted(os.listdir(path))
import PIL
import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import misc
import os
import imageio
from scipy import misc
#from keras.preprocessing import image
import scipy.stats

with open('E:/anchor_encodings_dict_facenet1.dat', 'rb') as f:
    all_face_encodings = pickle.load(f)


import itertools
import operator

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]


# In[2]:


# load the enco

# ding_network to make predictions based on trained network

# json_file = open("E:/encoding_network_arch_facenet.json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# encoding_network = model_from_json(loaded_model_json, custom_objects={'tf': tf,'epsilon':0.0000000000001})
encoding_network =kv.RESNET50(weights="vggface")
# load weights into new model
# encoding_network.load_weights('E:/encoding_network_weights_facenet.h5')
print("model loaded")


#%%
import time
#from imutils.video import VideoStream
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
capture = cv2.VideoCapture('http://192.168.1.3:8080/video')

NoneType =type(None)

# capture = cv2.VideoCapture(0)
time.sleep(3.0)
test = "E:/saved faces/"
kv.RESNET50()
from mtcnn.mtcnn import MTCNN
import time

person={0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"aghaei",7:"doctor",8:"saeed",9:"hemen"}

i=0
detector = MTCNN()
p = " "
percent=" "
name=p+percent
count=0
name2 = p
vote=[]
name3 = ""
t3=0
t1 = time.clock()
while (capture.isOpened()):

    ret, image = capture.read()

    if type(image)!=NoneType:

        image = cv2.resize(image, (1350, 735))
        x=336
        height, width, channels = image.shape
        upper_left = (int(width / 2)-int(x/2), int(height / 2)-int(x/2))
        bottom_right = (int(width / 2)+int(x/2), int(height / 2)+int(x/2))
        inner_image = image[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
        cv2.rectangle(image, upper_left, bottom_right, (0, 255, 0), 3)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = capture.frame
        cv2.imshow('video stream...', image)
        # image2=np.copy(image)
        #cv2.resizeWindow("frame",800,600)
        # inner_image = image[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
        # image2[0:, 0:upper_left[0] + 3] = 0
        # image2[0:, bottom_right[0] - 3:] = 0

        # image2[0:upper_left[1] + 3, 0:] = 0
        # image2[bottom_right[1] - 3:, 0:] = 0
        #cv2.imwrite(test+"1.png",inner_image)
        #image = frame
        #if cv2.waitKey(1) & 0xFF == 32:
        result = detector.detect_faces(inner_image)
        #print(i)
        print("Found {0} faces!".format(len(result)))
        if len(result) !=1:
            count=0
            name2 = ""
            vote=[]
            t1=time.clock()
            t2=time.clock()
            t3=t2-t1
            t3=int(t3*100)
            t3=t3/100

        if len(result) == 1:

            bounding_box = result[0]['box']
            cor = bounding_box
            cor[0]=cor[0]+upper_left[0]
            cor[1] = cor[1] + upper_left[1]

            if upper_left[1]>cor[1] or upper_left[0]>cor[0] or bottom_right[1]<cor[1]+cor[3] or bottom_right[0]<cor[0]+cor[2] or cor[2]<180 or cor[3]<180:
                cor=0
            # if cor[2]<224 or cor[3]<224:
            #     cor=0
            #keypoints = result[0]['keypoints']
            print("----------------")
            # print("Found {0} faces!".format(len(result)))

            # Draw a rectangle around the faces
            if cor == 0:
                count = 0
                name2 = ""
                vote = []
                t1 = time.clock()
                t2 = time.clock()
                t3 = t2 - t1
                t3 = int(t3 * 100)
                t3 = t3 / 100
                print("*** Face is not completely in square or you are too far from camera! ***\n*************************************")
            if cor!=0:
                cv2.rectangle(image, (cor[0], cor[1]), (cor[0]+cor[2], cor[1]+cor[3]), (0, 0, 255), 2)
                    #cor = [x, y, w, h]

                cv2.putText(image,name,(0,50),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 1)
                cv2.putText(image,name2,(cor[0],cor[1]-25),cv2.FONT_HERSHEY_SIMPLEX,0.75, (255,0 , 0), 2)
                cv2.putText(image,str(count),(cor[0],cor[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0 , 0,255), 1)
                cv2.putText(image, str(t3), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                cv2.imshow('video stream...', image)
                #cv2.imshow('frame', rect)
                #time.sleep(7.0)
                sub_face = image[cor[1] + 2:cor[1] + cor[3] - 2, cor[0] + 2:cor[0] + cor[2] - 2]

                if sub_face.shape[0]>0 and sub_face.shape[1]>0:
                    count = count + 1
                    t2=time.clock()
                    t3=t2-t1
                    t3=int( t3*100)
                    t3=t3/100
                    dim = (224, 224)
                    sub_face = cv2.resize(sub_face, dim,interpolation=cv2.INTER_AREA)
                    t=time.localtime()
                    address="face_{}_{}_{}_{}_{}_{}.png".format(t[0],t[1],t[2],t[3],t[4],t[5])
                    save_path = test + address
                    #cv2.imwrite(save_path, sub_face)
                    #anchor_img = Image.load_img(test + address, target_size=(224, 224))
                    #anchor_img= sub_face
                    anchor_img = sub_face / np.max(sub_face)
                    #print(np.mean(anchor_img), "aaaaa")
                    gray = color.rgb2gray(anchor_img)
                    sx = ndimage.sobel(gray, axis=0, mode='nearest', cval=0.0)
                    sy = ndimage.sobel(gray, axis=1, mode='nearest', cval=0.0)
                    sob = np.hypot(sx, sy)

                    # x = anchor_img.mean()
                    #x = sob.mean()
                    sob = sob / np.max(sob)
                    sob = np.repeat(sob[..., np.newaxis], 3, -1)
                    '''
                    sob[np.where(anchor_img == 1)] = 1
                    '''
                    im = 1 * sob + 1 * anchor_img
                    im = im / 2
                    im = im + 0.2
                    im = im / np.max(im)
                    cv2.imwrite(save_path, im*255)

                    #test_img = Image.load_img(save_path, target_size=(224, 224))
                    encoding_net_test_inputs = np.empty((0, 224, 224, 3))
                    test_img = Image.img_to_array(sub_face)
                    test_img = np.expand_dims(test_img, axis=0)
                    test_img = preprocess_input(test_img)
                    encoding_net_test_inputs = np.append(encoding_net_test_inputs, test_img, axis=0)
                    test_encoding = encoding_network.predict([encoding_net_test_inputs],
                                                             batch_size=1,
                                                             verbose=0)
                    row_dist = []
                    for (anchor_img_path, anchor_encoding) in all_face_encodings.items():
                        distance = np.linalg.norm(anchor_encoding - test_encoding)
                        row_dist.append(distance)

                    prob = np.maximum(0, (np.sqrt(2) - row_dist) * 100 / np.sqrt(2))
                    prob[prob < 1] = 0
                    prob = prob.astype("float16")
                    c=0
                    #select=[]
                    for i in prob:
                        print("\nprobability  {}".format(person[c])," = ",i,"%")

                        c=c+1
                    select=np.argmax(prob)

                    if prob[select]>85:
                        p = person[select]
                        percent=str(prob[select])
                        # print("\n=========================")
                        # print(" *** person = {}".format(p))
                        # print("=========================\n")
                        name = p + " ==>" + percent+"%"
                    else:
                        p = "Unknown"
                        # percent=" "
                        # print("\n=========================")
                        # print(" !!! ","{}".format(p)," !!!")
                        # print("=========================\n")
                        name = p
                    vote.append(p)
                    if(count==6):
                        v = most_common(vote)
                        name2 = v
                        name3=name2
                        vote=[]
                        count=0


                #name3 = name3   +"\n" + name +"\n"+str(count)
                    #time.sleep(1.0)
        #if cv2.waitKey(1) & 0xFF == ord("s"):
         #   cv2.imwrite(save_path, sub_face)

        #cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            capture.release()
            cv2.destroyAllWindows()
            break
    else:
        print("!!!! No Video streaming !!!!")
        break

capture.release()
cv2.destroyAllWindows()

#
# #%%
# x=124
# image3=cv2.imread("C:/Users/Erfan/Desktop/VaziyatSefid.jpg")
# image4=image3
# height, width, channels = image3.shape
# upper_left = (int(width / 2) - int(x / 2), int(height / 2) - int(x / 2))
# bottom_right = (int(width / 2) + int(x / 2), int(height / 2) + int(x / 2))
# cv2.rectangle(image3, upper_left, bottom_right, (0, 255, 255), 3)
# inner_image = image3[upper_left[1]+3: bottom_right[1]-3, upper_left[0]+3: bottom_right[0]-3]
# # image4[upper_left[1]+3: bottom_right[1]-2, upper_left[0]+3: bottom_right[0]-2]=255
# image4[0:,0:upper_left[0]+3]=0
# image4[0:,bottom_right[0]-3:]=0
#
# image4[0:upper_left[1]+3,0:]=0
# image4[bottom_right[1]-3:,0:]=0
#
# # image4[image4!=255]=0
# p=inner_image
# # p[:]=0
#
#
# cv2.imwrite("C:/Users/Erfan/Desktop/VaziyatSefid2.jpg",image4[:])