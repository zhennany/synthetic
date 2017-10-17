import numpy as np
import argparse
from imutils import face_utils
import imutils
import dlib
import cv2
import math
# import matplotlib.pyplot as plt
from scipy import misc
import scipy
from PIL import Image
import os
import glob
import shutil
from random import shuffle


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
	# help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
	# help="path to input image")
# ap.add_argument("-i2", "--image2", required=True,
	# help="path to input image2")
# ap.add_argument("-d", "--detect", type=int, default=1, required=False, help="whether run face detector first")
# args = vars(ap.parse_args())

args = {}
args['shape_predictor'] = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
dodetection = 1


def do_landmark(image, detector, predictor, dodetection, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    if dodetection:
        rects = detector(gray, 1)
        if len(rects) == 0:
            H,W = image.shape[0], image.shape[1]
            rects = [dlib.rectangle(0, 0, W, H)]
    else:
        H,W = image.shape[0], image.shape[1]
        rects = [dlib.rectangle(0, 0, W, H)]
        
    # assuming only one face should be detected
    assert len(rects) == 1
    rect = rects[0]
    
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
 
    if plot:
        timg = np.copy(image)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(timg, (x, y), (x + w, y + h), (0, 255, 0), 1)
     
     
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(timg, (x, y), 1, (0, 0, 255), -1)
     
        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", timg)
        cv2.waitKey(0)
    
        nose1 = shape[27:34]

        ang1 = nose_angle(nose1)
        M = cv2.getRotationMatrix2D((w/2,h/2),ang1,1)
        dst = cv2.warpAffine(timg,M,(w,h))
        cv2.imshow("Output", dst)
        cv2.waitKey(0)
        
    return shape
    
def blend_faces(image1, image2, lms1, lms2):
    
    m1_x = int(round(lms1[33,0]))
    m2_x = int(round(lms2[33,0]))
    
    H1,W1 = image1.shape[0], image1.shape[1]
    
    
    face_len1 = face_length(lms1)
    face_len2 = face_length(lms2)
    
    image2 = scipy.ndimage.interpolation.zoom(image2, (face_len1/face_len2, 1, 1))
    H2,W2 = image2.shape[0], image2.shape[1]
    lms2[:,1] = lms2[:,1]*face_len1/face_len2
    
    # left and right boundary
    r1_x = int(round(min(W1, max(lms1[:,0]) + 10)))
    r2_x = int(round(min(W2, max(lms2[:,0]) + 10)))
    l1_x = int(round(max(1, min(lms1[:,0]) - 10)))
    l2_x = int(round(max(1, min(lms2[:,0]) - 10)))
    
    # top and bottom boundary
    t1_y = int(round(max(1, min(lms1[:,1]) - 10)))
    t2_y = int(round(max(1, min(lms2[:,1]) - 10)))
    b1_y = int(round(min(H1, max(lms1[:,1]) + 10)))
    b2_y = int(round(min(H2, max(lms2[:,1]) + 10)))
    
    # new image
    newW = 2*(max((m1_x-l1_x), (r1_x-m1_x), (m2_x-l2_x), (r2_x-m2_x))+1)+10
    newH = max(H1, H2)
    newimage = np.zeros((newH, newW, 3))
    newimage2 = np.zeros((newH, newW, 3))
    grid_x = np.arange(newW)
    
    weight = 1/(1+np.exp(0.25*(grid_x-newW/2.)))
    
    # copy left image
    t_mx = newW/2
    w_1 = r1_x-l1_x+1
    t_x1 = t_mx- (m1_x-l1_x+1)
    t_y1 = (newH-H1)/2
    newimage[t_y1:t_y1+H1, t_x1:t_x1+w_1, :] = image1[:, l1_x-1:r1_x, :]
    
    newlms1 = lms1 + (t_x1-l1_x+1, t_y1)
    
    # import pdb
    # pdb.set_trace()
    # blend right half
    w_2 = r2_x -l2_x + 1
    diff = face_center(newlms1)-face_center(lms2)
    t_x2 = int(round(l2_x-1 + diff[0]))
    t_y2 = int(round(max(0, diff[1])))
    s_y2 = int(round(max(0, -diff[1])))
    h2 = min(H2-s_y2, newH-t_y2)

    newimage2[t_y2:t_y2+h2, t_x2:t_x2+w_2, :] = image2[s_y2:s_y2+h2, l2_x-1:r2_x, :]
    
    
    assert(newimage.shape == newimage2.shape)
    
    # misc.imsave('left.png', newimage)
    # misc.imsave('right.png', newimage2)
    # cv2.imshow('left', newimage)
    # cv2.imshow('right', newimage2)
    
    w = np.reshape(np.tile(np.transpose(np.tile(weight, (3, 1))), (newimage.shape[0], 1)), newimage.shape)
    newimage1 = w*newimage + (1-w)*newimage2
    newimage2 = (1-w)*newimage + w*newimage2
    
    return newimage1, newimage2
    
def stripBlackBoundary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H,W = gray.shape
    hist_x = np.sum(gray, axis=0)
    left = 0
    right = W
    for i in range(0, W):
        if hist_x[i] > 1:
            left = i
            break
    for i in range(W-1, 0, -1):
        if hist_x[i] > 1:
            right = i+1
            break

    x1 = min(W/3, W/2-20)-4
    x2 = max(W*2/3, W/2+20)+3
    y_left0 = 0
    y_left1 = H
    y_right0 = 0
    y_right1 = H
    for i in range(0, H):
        if gray[i, x1] > 1:
            y_left0 = i
            break
    for i in range(H-1, 0, -1):
        if gray[i, x1] > 1:
            y_left1 = i+1
            break
    for i in range(0, H):
        if gray[i, x2] > 1:
            y_right0 = i
            break
    for i in range(H-1, 0, -1):
        if gray[i, x2] > 1:
            y_right1 = i+1
            break
    top = max(y_left0, y_right0)
    bottom = min(y_left1, y_right1)
    
    return image[top:bottom, left:right, :]
    
    
for out_dir, new_out_dir in zip([r'F:\face\data_temp', r'F:\face\data_test_temp'], [r'face\train', r'face\test']):

    if not os.path.isdir(new_out_dir):
        os.makedirs(new_out_dir)
        os.mkdir(os.path.join(new_out_dir, '0'))
        os.mkdir(os.path.join(new_out_dir, '1'))

    ## class 0

    im_list = glob.glob(os.path.join(out_dir,'0','*.png'))
    for imfile in im_list:
        print imfile
        if 'yale' in imfile:
            image = cv2.imread(imfile)
            # image = imutils.resize(image, width=92)
            # H,W = image.shape[0], image.shape[1]
            # print H # 105
            misc.imsave(os.path.join(new_out_dir, '0', os.path.basename(imfile)), image)
        else:
            image = cv2.imread(imfile)
            H,W = image.shape[0], image.shape[1]
            lms = do_landmark(image, detector, predictor, dodetection, plot=False)
            y1 = min(H, max(lms[:,1])+3)
            y0 = max(0, min(lms[:,1])-3)
            h = abs(y1-y0)
            if h < 105:
                y1 = min(H, y1+(105-h+1)/2)
                y0 = max(0, y0-(105-abs(y1-y0)))
            # print y0, y1
            image = image[y0:y1,:,:]
            misc.imsave(os.path.join(new_out_dir, '0', os.path.basename(imfile)), image)

        
    ## class 1
    im_list = glob.glob(os.path.join(out_dir,'1','*.png'))
    for imfile in im_list:
        print imfile
        if 'yale' in imfile:
            image = cv2.imread(imfile)
            image = stripBlackBoundary(image)
            # image = imutils.resize(image, width=92)
            # H,W = image.shape[0], image.shape[1]
            # print H # 105
            misc.imsave(os.path.join(new_out_dir, '1', os.path.basename(imfile)), image)
        else:
            image = cv2.imread(imfile)
            # import pdb
            # pdb.set_trace()
            image = stripBlackBoundary(image)
            
            misc.imsave(os.path.join(new_out_dir, '1', os.path.basename(imfile)), image)
            
        