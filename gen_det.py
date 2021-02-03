import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
from glob import glob
import argparse

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def image_derivates(img) :
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.uint16)    
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],np.uint16) 
    Ix = cv2.filter2D(img,-1,sobel_x)
    Iy = cv2.filter2D(img,-1,sobel_x)
    return Ix,Iy

def satisfy(contours,shape):
    b = shape[1]
    h = shape[0]    
    area_min_threshold = 15000
    area_max_threshold = 100000
    M = cv2.moments(contours)
    area = cv2.contourArea(contours)
    if(area<area_min_threshold or area>area_max_threshold):
        return False
    if(M['m00']<=0):
        return False
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    if(cx < 0.25*b or cx > 0.75*b):
        return False;
    if(cy < 0.25*h or cy > 0.75*b):
        return False
    return True


def L2dist(x1,y1,x,y):
    return (x1-x)*(x1-x) + (y1-y)*(y1-y)

def getBest(list1,contours,shape):
    b = shape[1]
    h = shape[0]    
    if(len(list1)==1):
        return list1[0]
    if(len(list1)==0):
        return 0
    index = 0
    minValue = L2dist(b,h,0,0)
    for i in range(0,len(list1)):
        M = cv2.moments(contours[list1[i]])
        if(M['m00']>0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            l2norm = L2dist(cx,cy,b/2,h/2)
            if(l2norm < minValue):
                index = i;
                minValue = l2norm
    return list1[index]

def findMaskedImage(img):
    kernel = np.ones((5,5),np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    src = cv2.medianBlur(gray,5)
    src = cv2.GaussianBlur(src,(3,3),0)
    dst = gray
    thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 301, 32)
    thresh = cv2.erode(thresh,kernel,iterations = 2)
    thresh = cv2.dilate(thresh,kernel,iterations = 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    final = np.zeros(dst.shape)
    final1 = np.zeros(dst.shape)
    list1 = []
    for i in range(0,len(contours)):
        if(satisfy(contours[i],img.shape)):
            list1.append(i)
    best = getBest(list1,contours,img.shape)
    cv2.fillPoly(final,[contours[best]], [255,255,255])
    final = cv2.dilate(final,kernel,iterations = 1)
    return final,cv2.drawContours(final1,contours,-1,(255,255,255),3),thresh

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--img_path",metavar="DIR",help="path to original image dir",type=str)
	parser.add_argument("--det_path",metavar="DIR",help="path to original image dir",type=str)
	args = parser.parse_args()
	current_directory = os.getcwd()
	img_path = os.path.relpath(args.img_path, start = os.curdir)
	det_path = os.path.relpath(args.det_path, start = os.curdir)
	if not os.path.isdir(det_path):
		os.makedirs(det_path)
	current_directory = os.getcwd() 
	img_files = sorted(glob(os.path.join(img_path, "*jpg")))
	os.chdir(img_path)
	img_names = sorted(glob(os.path.join("", "*jpg")))
	os.chdir(current_directory)
	for fimg,fname in zip(img_files,img_names):
	    img = cv2.imread(fimg)
	    final,masked,thresh = findMaskedImage(img)
	    os.chdir(det_path)
	    cv2.imwrite(fname,final)
	    os.chdir(current_directory)
