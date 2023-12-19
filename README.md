# Pytorch Implementation of CVit-Net: A Conformer Driven RGB-D Salient Object Detector with Operation-Wise Attention Learning, 
authors:"Samra Kanwal and Imtiaz Ahmad Taj"

# Requirements
* Python 3.6 <br>
* Pytorch 1.5.0 <br>
* Torchvision 0.6.1 <br>
* Cuda 11.0

# Usage
This is the Pytorch implementation of  CVit-Net: A Conformer Driven RGB-D Salient Object Detector with Operation-Wise Attention Learning  It has been trained and tested on Linux (Cuda 11 + Python 3.6 + Pytorch 1.5), and it should also work on Windows but we didn't try. 

## To Train 
* Download the pre-trained ImageNet [backbone](#pre\-trained-imagenet-model-for-training) (conformerB), and put it in the 'pretrained' folder.
* Download the [training dataset](#dataset) and modify the 'train_root' and 'train_list' in the `main.py`.

* Start to train with
python main.py

## To Test 
* Download the [testing dataset](#dataset) and have it in the 'dataset/test/' folder. 
* Download the [already-trained CVit-Net pytorch model](#trained-model-for-testing) and modify the 'model' to its saving path in the `main.py`.
* Start to test with
python main.py --mode=test --sal_mode=STERE --test_root=xx/STERE --test_list=xx/STERE/test.lst --test_folder=xx/test_r/STERE --model=./checkpoints/demo-07/final.pth  --batch_size=1

# Pre-trained ImageNet model for training
Google Drive: -------------- <br>

# Dataset
Baidu Pan:<br>
[Training dataset (with horizontal flip)](https://pan.baidu.com/s/1vrVcRFTMRO5v-A6Q2Y3-Nw), password:  i4mi<br>
[Testing datadet](https://pan.baidu.com/s/13P-f3WbA76NVtRePcFbVFw), password:   1ju8<br>
Google Drive:<br>
[Training dataset (with horizontal flip)](https://drive.google.com/open?id=12ais7wZhTjaFO4BHJyYyNuzzM312EWCT)<br>
[Testing datadet](https://drive.google.com/open?id=18ALe_HBuNjVTB_US808d8ZKfpd_mwLy5)<br>
