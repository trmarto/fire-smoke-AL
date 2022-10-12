from pyexpat import model
from keras import backend as K
import numpy as np
import cv2
import re
import pandas as pd
import os
import tensorflow as tf
import keras

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def cal_miou(pred_mask, sample_mask):
    tp = np.sum(cv2.bitwise_and(pred_mask, sample_mask))
    fp = np.sum(cv2.bitwise_and(pred_mask, cv2.bitwise_not(sample_mask)))
    fn = np.sum(cv2.bitwise_and(cv2.bitwise_not(pred_mask), sample_mask))
    return tp/(tp+fp+fn)

'''
def pixelAccuracy(y_pred, y_true):
    y_pred = np.argmax(np.reshape(y_pred,[2,y_pred.shape[0],y_pred.shape[1]]),axis=0)
    y_true = np.argmax(np.reshape(y_true,[2,y_true.shape[0],y_true.shape[1]]),axis=0)
    y_pred = y_pred * (y_true>0)

    return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)
'''



def pixelAccuracy(imPred, imLab):
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled

    return pixel_accuracy


img_path_al = "../../../ciafa/mnt_point_3/trmarto/files/data/segmentation/fire_al/"
img_path_gt = "../../../ciafa/mnt_point_3/trmarto/files/data/segmentation/fire_gt/"


images_al = sorted_alphanumeric(os.listdir(img_path_al)) 
images_gt = sorted_alphanumeric(os.listdir(img_path_gt)) 


#img_al = "072_al.png"
#img_gt = "072_gt.png"
values_iou = []
values_pixel_accuracy = []

for img_al, img_gt in zip(images_al, images_gt):
  if img_al[:-6] != img_gt [:-6]:
    raise("error")
  values_iou.append(cal_miou(cv2.imread(img_path_al + img_al), cv2.imread(img_path_gt + img_gt)))
  values_pixel_accuracy.append(pixelAccuracy(cv2.imread(img_path_al + img_al), cv2.imread(img_path_gt + img_gt)))

print(np.mean(np.array(values_iou)))
print(np.mean(np.array(values_pixel_accuracy)))

#img_al = cv2.imread(img_path + "fire_al/" + img_al)
#img_gt = cv2.imread(img_path + "fire_gt/" + img_gt)
#print(img_al.shape)
#print(img_gt.shape)
'''




'''
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou



def computeIoU(y_pred_batch, y_true_batch):
    return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))])) 


    