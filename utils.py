import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras.layers import  Conv3D, Activation, MaxPooling3D, Conv3DTranspose, Add,BatchNormalization, Dropout
from tensorflow.keras.callbacks import  ModelCheckpoint, TensorBoard
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import math
from glob import glob
from random import shuffle, randint
#import random
#from metrics import *
from numpy import linalg as LA
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nibabel as nib
import medpy.metric.binary as medpy_metrics 
from numpy import linalg as LA
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from datetime import datetime
from skimage import io

#import sys
#sys.settrace
from nibabel import load as load_nii

#from scipy.stats import boxcox



def save_model(modelInput, modelNameInput='model'):
	model_json = modelInput.to_json()
	model_json = json.loads(model_json)
	model_json['class_name'] = 'Model' # this attribute sometimes is not properly set
	model_json = json.dumps(model_json)
	with open(modelNameInput+".json","w") as json_file:
		json_file.write(model_json)
	modelInput.save_weights(modelNameInput+".h5")
	print("Saved "+modelNameInput)

def load_model(modelNameInput = 'model'):
	json_file = open(modelNameInput+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(modelNameInput+".h5")
	return loaded_model
	print("Loaded model from disk")





def ensure_dir(directory):
  if not os.path.exists(directory):
      os.makedirs(directory)



def add_padding_z(img,depth_with_padding):
    pad_z = depth_with_padding
    pad_value=img[0][0][0]

    image_padded = np.empty((img.shape[0],img.shape[1],pad_z))
    image_padded.fill(pad_value)
    image_padded[:,:,np.floor_divide((depth_with_padding-img.shape[2]),2):-math.ceil((depth_with_padding-img.shape[2])/2)] = img
    return image_padded

def add_padding_x(img,depth_with_padding):
    pad_x = depth_with_padding
    
    pad_value=img[0][0][0]

    image_padded = np.empty((pad_x,img.shape[1],img.shape[2]))
    image_padded.fill(pad_value)
    image_padded[np.floor_divide((depth_with_padding-img.shape[0]),2):-math.ceil((depth_with_padding-img.shape[0])/2),:,:] = img
    return image_padded

def add_padding_y(img,depth_with_padding):
    pad_y = depth_with_padding
   
    pad_value=img[0][0][0]

    image_padded = np.empty((img.shape[0],pad_y,img.shape[2]))
    image_padded.fill(pad_value)
    image_padded[:,np.floor_divide((depth_with_padding-img.shape[1]),2):-math.ceil((depth_with_padding-img.shape[1])/2),:] = img
    return image_padded







def multi_class_prediction(prediction):
    prediction_images = []
    for i in range(prediction.shape[4]):
        prediction_images.append(nib.Nifti1Image(prediction[0,:,:,:, i],None))


def one_hot_labels(data, n_labels, labels=None):

    new_shape = [data.shape[0] , data.shape[1], data.shape[2], data.shape[3], n_labels]
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, :,:,:,label_index][data[:,:,:,:, 0] == labels[label_index]] = 1

    return y

def set2to0(matrix):
	matrix[matrix>1.0] = 0.0
	return matrix

