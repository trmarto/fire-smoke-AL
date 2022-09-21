import os
import re
import cv2
import tensorflow as tf 
import keras
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from keras import *
from keras import backend as K
from pydensecrf.utils import  unary_from_labels


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def get_output_layer(model, layer_name):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def crf_dense(maskd,image_rgb,r,xy):
    n_classes = 2
    colors, labels = np.unique(maskd.flatten(), return_inverse=True)
    unary = unary_from_labels(labels, 2, 0.7,zero_unsure = False)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(image_rgb.shape[1], image_rgb.shape[0], n_classes)
    
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(10), compat=5, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=(xy), srgb=(r), rgbim=image_rgb,
                        compat=8,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
      
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((image_rgb.shape[0], image_rgb.shape[1]))
    res_hot = utils.to_categorical(res) * 255.0
    crf_mask = np.array(res*255, dtype=np.uint8)

    return crf_mask

    
def get_output(folder_path, model_name, path_test):

    images_np = sorted_alphanumeric(os.listdir(path_test)) 
    K.clear_session()
    model = models.load_model("models/"+model_name)
    # model.summary()

    out_class = 1
    out_mask_list = []

    for img in images_np:
        img_path = path_test+img
        img_name = img
        print("\nImg processing: " , img_name)
        original_img = cv2.imread(img_path)
        width, height, _ = original_img.shape

        img_o = preprocessing.image.load_img(img_path, target_size=(256, 256))
        img_o = preprocessing.image.img_to_array(img_o)
        img_o = np.expand_dims(img_o, axis=0)
        input_img = applications.vgg16.preprocess_input(img_o)     

        # Image perdiction
        preds = model.predict(input_img)
        # Return empty mask if no smoke in image
        if preds[0][out_class] < 0.7 :
            out_mask_list.append(np.zeros((height, width), dtype=float))
            continue

        #- - - - - - - - - CAM - - - - - - - - -#

        # Get the 512 input weights to the sigmoid.
        class_weights = model.layers[-1].get_weights()[0]
        
        # Get the outputs of the last conv layer
        final_conv_layer = get_output_layer(model, "block5_conv3")
        get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
        [conv_outputs, predictions] = get_output([input_img])
        conv_outputs = conv_outputs[0, :, :, :]

        #Create the Class Activation Mapping
        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
    
        for p, w in enumerate(class_weights[:, out_class]):
            cam += w * conv_outputs[:, :, p]

        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
    
        # Heatmap image
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.1)] = 0
        output_heatmap = folder_path+"compare/heatmap/"+img_name
        # cv2.imwrite(folder_path+"compare/cam_w0/img/"+img_name+'_mask.png',heatmap)

        # Original image with heatmap overlayed
        img_heat = heatmap*0.8 + original_img
        output_img_heat = folder_path+"compare/heatmap_over/"+img_name+'_out.png'
        # cv2.imwrite(output, img_heat)

        # Convert heatmap into binary mask        
        alpha = 0.2
        tresh = alpha * np.max(cam)

        # Binary mask (heatmap segmented)
        heatmap_seg = np.uint8(cam)
        heatmap_seg[np.where(cam < tresh)] = 0
        heatmap_seg[np.where(cam >= tresh)] = 1
        output_seg2 = folder_path+"compare/bin_mask/"+img_name+'_cam_mask.png'
        #cv2.imwrite(output_seg2, 255*heatmap_seg)

        # Original image with binary mask overlayed
        heatmap_img_seg = cv2.cvtColor(heatmap_seg ,cv2.COLOR_GRAY2RGB)
        img_mask = original_img + heatmap_img_seg*0.3*255
        output_segimg2 = folder_path+"compare/img_w_mask/"+img_name+'_seg_img.png'
        cv2.imwrite(output_segimg2, img_mask)

        #- - - - - - - - - CRF - - - - - - - - -#

        image_rgb = cv2.resize(original_img,(500,500))
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        
        cam_mask = cv2.resize(heatmap_seg,(500,500))
        mask_inv = cv2.bitwise_not(cam_mask)
        maskd = np.expand_dims(cam_mask, axis=2)
        mask_inv = np.expand_dims(mask_inv, axis=2)
        
        # CRF parameters
        r = 100
        xy = 5

        crf_mask = crf_dense(maskd,image_rgb,r,xy)
        crf_mask = cv2.resize(crf_mask, (height, width))
        out_mask_list.append(crf_mask)

    return out_mask_list


def main():
    folder_path = "models/"
    path_test = "../data/fire_test/"

    model_name = "smoke_model.h5"

    output_mask_list = get_output(folder_path, model_name, path_test)


if __name__ == '__main__':
    main()