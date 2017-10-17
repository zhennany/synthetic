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

args['detect'] = 1

def show_landmark(image, lm):
    timg = np.copy(image)
    for (x, y) in lm:
        cv2.circle(timg, (x, y), 1, (0, 0, 255), -1)
 
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", timg)
    cv2.waitKey(0)

    
def do_landmark(image, detector, predictor, dodetection, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    if dodetection:
        rects = detector(gray, 1)
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
    
def nose_angle(nose_lm):
    x1,y1 = nose_lm[0]
    x2,y2 = nose_lm[-1]
    return -180* math.atan2((x2-x1) , abs(y2-y1)) / math.pi
    
def align_face_nose(image, lms):
    H,W = image.shape[0], image.shape[1]
    nose1 = lms[27:34] # lms[27:31]

    ang1 = nose_angle(nose1)

    M = cv2.getRotationMatrix2D((W/2,H/2),ang1,1)
    dst = cv2.warpAffine(image,M,(W,H))
    
    tlms = np.concatenate((lms, np.ones((68,1))), axis=1)
    newlms = np.transpose(np.dot(M, tlms.T)).astype(np.int)
    
    return dst, newlms, M

def face_length(lms):
    eye_y = float(lms[39,1]+lms[42,1])/2.
    mouth_y = float(lms[62,1]+lms[66,1])/2.
    return abs(mouth_y-eye_y)
    
def face_center(lms):
    # return lms[33,:] # nose tip
    return np.mean(lms[27:, :], axis=0)
    
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
    
    
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
dodetection = int(args["detect"]) == 1

org_root_dir = r'F:\face\att'

root_dir = r'F:\face\my_collection' # manually selected frontal faces, 'm': male, 'mg': male with glasses, 'f': female, 'fg': female with glasses

out_dir = r'F:\face\data_temp'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
    os.mkdir(os.path.join(out_dir, '0'))
    os.mkdir(os.path.join(out_dir, '1'))

out_dir_delib = r'F:\face\data_test_temp'
if not os.path.isdir(out_dir_delib):
    os.makedirs(out_dir_delib)
    os.mkdir(os.path.join(out_dir_delib, '0'))
    os.mkdir(os.path.join(out_dir_delib, '1'))


for subdir in ['m', 'mg', 'f', 'fg']:
    im_list = glob.glob(os.path.join(root_dir,subdir,'*.pgm'))
    if len(im_list) == 0:
        continue
    im_list.sort(key = lambda x: int(os.path.basename(x)[1:os.path.basename(x).rindex('.')])) # sort with filename (id), which start from 's'
    
    for i in range(len(im_list)):
        # shutil.copy(im_list[i], os.path.join(out_dir, '0', subdir+str(i)+'.pgm'))
        
        image = cv2.imread(im_list[i])
        misc.imsave(os.path.join(out_dir, '0', subdir+str(i)+'.png'), image)
        
        
        for j in range(i+1, len(im_list)):
            
            image2 = cv2.imread(im_list[j])

            lm1_file = im_list[i][:-len('.pgm')]+'_lm.npy'
            if os.path.isfile(lm1_file):
                lms1 = np.load(lm1_file)
            else:
                lms1 = do_landmark(image, detector, predictor, dodetection, plot=False)
            
            lm2_file = im_list[j][:-len('.pgm')]+'_lm.npy'
            if os.path.isfile(lm2_file):
                lms2 = np.load(lm2_file)
            else:
                lms2 = do_landmark(image2, detector, predictor, dodetection, plot=False)

            aimg1, lms1, M1 = align_face_nose(image, lms1)
            aimg2, lms2, M2 = align_face_nose(image2, lms2)

            bimage1, bimage2 = blend_faces(aimg1, aimg2, lms1, lms2)

            misc.imsave(os.path.join(out_dir, '1', subdir+'%d_%d.png'%(i,j)), bimage1)

            misc.imsave(os.path.join(out_dir_delib, '1', subdir+'%d_%d.png'%(j,i)), bimage2)
            
    
## additional class0 samples, have head pose variations
subdirs = os.listdir(org_root_dir)

for subdir in subdirs:
    im_list = glob.glob(os.path.join(org_root_dir,subdir,'*.pgm'))
    count = len(im_list)
    if count == 0:
        continue
    x = [i for i in range(count)]
    shuffle(x)
    im_list = [im_list[i] for i in x]
    
    for i in range(count/2):
        image = cv2.imread(im_list[i])
        f = os.path.basename(im_list[i])
        misc.imsave(os.path.join(out_dir, '0', subdir+'_'+f[:-4]+'.png'), image)
    for i in range(count/2, count):
        image = cv2.imread(im_list[i])
        f = os.path.basename(im_list[i])
        misc.imsave(os.path.join(out_dir_delib, '0', subdir+'_'+f[:-4]+'.png'), image)
        

