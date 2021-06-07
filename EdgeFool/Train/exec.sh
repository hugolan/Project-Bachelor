#!/bin/bash

PATH_IMGS=(../Dataset/)

PATH_IMGS_ADV=(../EnhancedAdvImgsfor_resnet50/)


python -W ignore test.py --path_imgs=$PATH_IMGS --path_imgs_adv=$PATH_IMGS_ADV