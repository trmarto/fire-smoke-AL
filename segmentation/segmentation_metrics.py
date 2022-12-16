from optparse import Values
from pyexpat import model
from keras import backend as K
import numpy as np
import cv2
import re
import pandas as pd
import os
import sys
import json

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def cal_miou(pred_mask, sample_mask):
    tp1 = np.sum(cv2.bitwise_and(pred_mask, sample_mask))
    fp1 = np.sum(cv2.bitwise_and(pred_mask, cv2.bitwise_not(sample_mask)))
    fn1 = np.sum(cv2.bitwise_and(cv2.bitwise_not(pred_mask), sample_mask))

    tp0 = np.sum(cv2.bitwise_and(cv2.bitwise_not(pred_mask), cv2.bitwise_not(sample_mask)))
    fp0 = np.sum(cv2.bitwise_and(cv2.bitwise_not(pred_mask), sample_mask))
    fn0 = np.sum(cv2.bitwise_and(pred_mask, cv2.bitwise_not(sample_mask)))

    class0 = tp0/(tp0+fp0+fn0 + 0.0000000000001)
    class1 = tp1/(tp1+fp1+fn1 + 0.0000000000001)

    return np.mean([class0,class1])


def pixelAccuracy(imPred, imLab):
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled

    return pixel_accuracy


    
def main(argv):
  
  if argv[0] == "F":
    img_path_al = "../../../ciafa/mnt_point_3/trmarto/files/data/segmentation/" + argv[1][:-3] + "/"
    outfile = "segmentation_data_fire.txt"
    img_path_gt = "../../../ciafa/mnt_point_3/trmarto/files/data/segmentation/fire_gt/"
  elif argv[0] == "S":
    img_path_al = "../../../ciafa/mnt_point_3/trmarto/files/data/segmentation/" + argv[1][:-3] + "/"
    outfile = "segmentation_data_smoke.txt"
    img_path_gt = "../../../ciafa/mnt_point_3/trmarto/files/data/segmentation/smoke_gt/"
  

  images_al = sorted_alphanumeric(os.listdir(img_path_al)) 
  images_gt = sorted_alphanumeric(os.listdir(img_path_gt)) 


  values_iou = []
  values_pixel_accuracy = []
  dict_list_aux = []
  results_save ={}



  for img_al, img_gt in zip(images_al, images_gt):
    if argv[0] == "S":
      if img_al[:-7] != img_gt[:-4]:
        raise("error")
      values_iou.append(cal_miou(cv2.imread(img_path_al + img_al), cv2.imread(img_path_gt + img_gt)))
      values_pixel_accuracy.append(pixelAccuracy(cv2.imread(img_path_al + img_al), cv2.imread(img_path_gt + img_gt)))
    elif argv[0] == "F":
      if img_al[:-6] != img_gt[:-6]:
        raise("error")
      values_iou.append(cal_miou(cv2.imread(img_path_al + img_al), cv2.imread(img_path_gt + img_gt)))
      values_pixel_accuracy.append(pixelAccuracy(cv2.imread(img_path_al + img_al), cv2.imread(img_path_gt + img_gt)))

  dict_list_aux.append(np.mean(np.array(values_iou)))
  dict_list_aux.append(np.mean(np.array(values_pixel_accuracy)))
  results_save[str(argv[1])] = dict_list_aux 


  with open(outfile, 'a') as file:
     file.write(json.dumps(results_save))


if __name__ == '__main__':
    main(sys.argv[1:])