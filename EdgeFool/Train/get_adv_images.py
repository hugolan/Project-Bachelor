#! /usr/bin/python
import argparse

import copy
import argparse
import os, sys
import numpy
import os
import time
import copy
import torch
from os.path import join,isfile
from tqdm import tqdm
from os import listdir
import random
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import autograd
from utils import Config
from dataset import SuDataset
from vis_utils import VisUtils
from module import DeepGuidedFilter
import numpy as np
import cv2
import torchvision.transforms as T
from torchvision import models
from torch.nn import functional as F
from misc_function import processImage, detail_enhance_lab, recreate_image, PreidictLabel, AdvLoss

config = Config(
	N_START = 0,
	N_EPOCH = None,
	FINE_SIZE = -1,
	#################### CONSTANT #####################
	IMG = None,
	SAVE = 'checkpoints',
	BATCH = 1l,
	GPU = 0,
	LR = 0.001,
	# clip
	clip = None,
	# model
	model = DeepGuidedFilter(),
	# forward
	forward = None,
	# img size
	exceed_limit = None,
	# vis
	vis = None
)

print("Started")

parser = argparse.ArgumentParser(description='Test class')
parser.add_argument('--path_imgs', required=True, nargs='+', type=str,
            help='a list of image paths, or a directory name')
parser.add_argument('--path_imgs_adv', required=True, nargs='+', type=str,
            help='a list of image paths, or a directory name')
args = parser.parse_args()

#image path
dataset_path = args.path_imgs[0] 

#adversarial image path
dataset_adversarial_path = args.path_imgs_adv[0] 

# Smoothing loss function
criterion = nn.MSELoss()

# Using GPU
#if config.GPU >= 0:
#	with torch.cuda.device(config.GPU):
#		config.model.cuda()
#		criterion.cuda()

#load existing state

config.model.load_state_dict(torch.load('/home/hugo/Desktop/EdgeFool/Train/checkpoints/snapshots/resnet50_latest.pth'))

#image list for images without modifying
image_list =  [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path,f))]

#image list for images with modifications
adversarial_image_list =  [f for f in os.listdir(dataset_adversarial_path) if os.path.isfile(os.path.join(dataset_adversarial_path,f))]

NumImg=len(image_list)



#classifier
classifier = models.resnet50(pretrained=True)
classifier.cuda()
classifier.eval()


#number of images correctly predicted
correctly_missclassified = 0



#list of (image label,adversarial image label)
labels = []



# Prediction of the original image using the classifier chosen for attacking
for image_number in range(NumImg):


    x= processImage(dataset_path,img_name)		
	gt_smooth = processImage(dataset_smooth_path,img_name)
    
	
    # Smooth images
	x_smooth= config.forward(x,gt_smooth, config)
    # Enhance adversarial image
	enh = detail_enhance_lab(x,x_smooth)
	   

    #adv img
    adv_path =	'../adv_images_test/'
    cv2.imwrite('{}{}'.format(radv_path,img_name), recreate_image(enh))



print("ended")