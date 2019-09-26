#%%
import cv2
import os
import sys
#%%
# Get user supplied values

#cascPath = "E:/face_siamese/Preprocess/haarcascade_frontalface_alt2.xml"

'''
directory="E:/face_siamese/Towsan_final/Towsan_final/train/"
directory2="E:/face_siamese/Towsan_final/Towsan_final/train1/"
files = sorted(os.listdir(directory2))
'''

#test2="E:/face_siamese/Towsan_final/Towsan_final/test wo detect/"
#test="E:/face_siamese/Towsan_final/Towsan_final/test detected new/"

#test2="C:/Users/Erfan/Desktop/New folder (4)/"
#test= "C:/Users/Erfan/Desktop/New folder (5)/"

test2 = "C:/Users/Erfan/Desktop/New folder/"
test = "C:/Users/Erfan/Desktop/New folder1/"

files2 = sorted(os.listdir(test2))
#del(files2[1])
#%%
count=1
for i in files2:

    print(count," of ",len(files2))
    count=count+1
    address = i
    imagePath =test2+address
    # Create the haar cascade
    #faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    '''
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=9,
        minSize=(20, 20)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    '''
    from mtcnn.mtcnn import MTCNN

    detector = MTCNN()
    result = detector.detect_faces(image)
    print(i)
    print("Found {0} faces!".format(len(result)))
    if len(result)>0:
        bounding_box = result[0]['box']
        cor=bounding_box
        keypoints = result[0]['keypoints']
        print("----------------")
        #print("Found {0} faces!".format(len(result)))

        # Draw a rectangle around the faces
        '''
        for (x, y, w, h) in result:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cor = [x, y, w, h]
        '''
        sub_face = image[cor[1] + 2:cor[1] + cor[3] - 2, cor[0] + 2:cor[0] + cor[2] - 2]


        dim = (224, 224)
        sub_face = cv2.resize(sub_face, dim, interpolation=cv2.INTER_AREA)
        #cv2.imwrite(p, sub_face)
        #total += 1


        save_path=test+address
        #cv2.imshow("face",sub_face)
        #cv2.waitKey(0)
        #cv2.waitKey(0)
        #cv2.waitKey(0)
        cv2.imwrite(save_path,sub_face)


    else:
        print("*!*!!*!*!!*!*!*!*!*!*!")
        save_path=test+"No face_"+address
        cv2.imwrite(save_path, image)


