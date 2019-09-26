#%%
import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import skimage
"""
if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()
"""
#predictor_path =
#faces_folder_path = sys.argv[2]
predictor_path = "E:/face_siamese/models/Preprocess/shape_predictor_68_face_landmarks.dat"
faces_folder_path = "E:/face_siamese/Towsan_final/test1/"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#%%
win = dlib.image_window()

directory="C:/Users/Erfan/Desktop/New folder1/"
directory2="C:/Users/Erfan/Desktop/New folder/"
files = sorted(os.listdir(directory2))



for f in files:
    print("Processing file: {}".format(f))
    #img = dlib.load_rgb_image(f)
    img=io.imread(directory2+f)
    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)

    #win.add_overlay(dets)

    #input("Press Enter to continue")

    #dlib.hit_enter_to_continue()


    p=[]
    for i in range(0,17):
        z=[shape.part(i).x],[shape.part(i).y]
        p=np.append(p,z)
    #p=p.reshape(17,2)
    #p=np.array(p)
    x=[]
    y=[]
    h=[]
    for i in range(17,27):
        y=np.append(y,shape.part(i).y)
        x=np.append(x,shape.part(i).x)

    x=np.flip(x)
    y=np.flip(y)
    for i in range(10):
        h=np.append(h,(x[i],y[i]))

    g=[]
    p=np.append(p,h)
    for i in range(0,53,2):
        g.append((p[i],p[i+1]))
    face=np.copy(g)
    face=face.reshape(1,27,2)





    v=[15,30,50,60,70,70,60,50,30,15]
    for u in range(17,27):
        face[0][u][1]=face[0][u][1]-v[u-17]
        if face[0][u][1]<0:
            face[0][u][1]=0

    face[0][0][0]=face[0][0][0]-5
    face[0][16][0] = face[0][16][0] + 5
    if face[0][0][0] < 0:
        face[0][0][0] = 0
    if face[0][16][0] > 223:
        face[0][16][0] = 223



    face=face.astype("int32")

    z=[]
    left_eye=[]
    for i in range(36,42):
        z=[shape.part(i).x],[shape.part(i).y]
        left_eye=np.append(left_eye,z)
    z=[]
    right_eye=[]
    for i in range(42,48):
        z=[shape.part(i).x],[shape.part(i).y]
        right_eye=np.append(right_eye,z)

    left_eye =left_eye.reshape(1,6,2)
    left_eye=left_eye.astype("int32")

    right_eye=right_eye.reshape(1,6,2)
    right_eye=right_eye.astype("int32")


    #left_eye=list(left_eye)
    #right_eye=list(right_eye)

    import numpy
    from PIL import Image, ImageDraw
    # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]

    import cv2
    import numpy as np

    # load the image
    #image_path = "E:/face_siamese/Preprocess/source/a39ca49663d8c3f36d0755ae27786dd9.jpg"
    # io.imsave(directory + f, face)
    #image = cv2.imread(image_path)
    face=face.astype("int32")
    # create a mask with white pixels
    mask_face = np.ones(img.shape, dtype=np.uint8)
    mask_face.fill(255)

    mask_left_eye = np.ones(img.shape, dtype=np.uint8)
    mask_left_eye.fill(255)

    mask_right_eye = np.ones(img.shape, dtype=np.uint8)
    mask_right_eye.fill(255)

    # points to be cropped
    #roi_corners = np.array([[(0, 300), (1880, 300), (1880, 400), (0, 400),(0,500)]], dtype=np.int32)
    # fill the ROI into the mask
    cv2.fillPoly(mask_face,face , (0,0,0))

    cv2.fillPoly(mask_left_eye,left_eye , 0)

    cv2.fillPoly(mask_right_eye,right_eye , 0)

    # The mask image
    cv2.imwrite('image_masked_f.png', mask_face)

    cv2.imwrite('image_masked_l.png', mask_left_eye)

    cv2.imwrite('image_masked_r.png', mask_right_eye)

    # applying th mask to original image
    masked_image_face = cv2.bitwise_or(img, mask_face)
    #masked_image_face[np.where(mask_left_eye==0)]=255
    #masked_image_face[np.where(mask_right_eye==0)]=255
    # The resultant image
    #e=np.where(img == 255)
    #b=np.where(mask_face==255)
    #w=e or b
    #img[w]=0
    masked_image_face=masked_image_face+0.5*mask_face
    masked_image_face=masked_image_face/np.max(masked_image_face)
    masked_image_face[np.where(masked_image_face==1)]=0
    io.imsave(directory+f, masked_image_face)



#%%
win = dlib.image_window()

directory="C:/Users/Erfan/Desktop/New folder1/"
directory2="C:/Users/Erfan/Desktop/New folder/"
files = sorted(os.listdir(directory2))



for f in files:
    print("Processing file: {}".format(f))
    #img = dlib.load_rgb_image(f)
    img=io.imread(directory2+f)
    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)

    #win.add_overlay(dets)

    #input("Press Enter to continue")

    #dlib.hit_enter_to_continue()



    io.imsave(directory+f, masked_image_face)

