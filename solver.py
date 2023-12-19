import torch
from torch.nn import functional as F
from conformer import build_model
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
import torch.nn as nn
import argparse
import os.path as osp
import os
size_coarse = (10, 10)
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from utils import count_model_flops
from PIL import Image
import json



class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        #self.build_model()
        self.net = build_model(self.config.network, self.config.arch)
        #self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model for testing from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        if config.mode == 'train':
            if self.config.load == '':
                print("Loading pre-trained imagenet weights for fine tuning")
                self.net.JLModule.load_pretrained_model(self.config.pretrained_model
                                                        if isinstance(self.config.pretrained_model, str)
                                                        else self.config.pretrained_model[self.config.network])
                # load pretrained backbone
            else:
                print('Loading pretrained model to resume training')
                self.net.load_state_dict(torch.load(self.config.load))  # load pretrained model
        
        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'Conformer based SOD Structure')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params_t = 0
        num_params=0
        param_size = 0

        for p in model.parameters():
            param_size += p.nelement() * p.element_size()
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        #print(name)
        #print(model)
        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.6f}MB'.format(size_all_mb))
        print("The number of trainable parameters: {:.6f}".format(num_params_t))
        print("The number of parameters: {:.6f}".format(num_params))
        print(f'Flops:{count_model_flops(model)}')

    # build the network
    '''def build_model(self):
        self.net = build_model(self.config.network, self.config.arch)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'JL-DCF Structure')'''

    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, depth, name = data_batch['image'], data_batch['depth'], data_batch['name'][0]
                                          
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                #input = torch.cat((images, depth), dim=0)
                # start time
                torch.cuda.synchronize()
                tsince = int(round(time.time()*1000)) 
                preds,coarse_sal_rgb,coarse_sal_depth,f_att3,f_att2,corr_rgb2d2,corr_rgb2d3,corr_d2rgb2,corr_d2rgb3,Att,e_rgbd0,e_rgbd1,e_rgbd2,rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,depth_1,depth_2,depth_3,depth_4,depth_5,rgbd_fusion_1,rgbd_fusion_2,rgbd_fusion_3,rgbd_fusion_4,rgbd_fusion_5= self.net(images,depth)
                torch.cuda.synchronize()
                ttime_elapsed = int(round(time.time()*1000)) - tsince
                print ('test time elapsed {}ms'.format(ttime_elapsed))
                preds = F.interpolate(preds, tuple(320), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()
                #print(pred.shape)
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_convtran.png')
                cv2.imwrite(filename, multi_fuse)
                '''f_att3=(torch.sum(f_att3,1)/f_att3.shape[1]).unsqueeze(0)
                #f_att3 = f_att3[0].clone()
                f_att3 = F.interpolate(f_att3, tuple(im_size), mode='bilinear', align_corners=True)
                f_att3 = f_att3.sigmoid().data.cpu().numpy().squeeze()
                f_att3 = (f_att3 -f_att3.min()) / (f_att3.max() - f_att3.min() + 1e-8)
                f_att3 = 255 * f_att3
                filename = os.path.join(self.config.test_folder, name[:-4] + 'f_att3_rgbonly.png')
                cv2.imwrite(filename, f_att3)
                f_att2=(torch.sum(f_att2,1)/f_att2.shape[1]).unsqueeze(0)
                #f_att2 = f_att2[0].clone()
                f_att2 = F.interpolate(f_att2, tuple(im_size), mode='bilinear', align_corners=True)
                f_att2 = f_att2.sigmoid().data.cpu().numpy().squeeze()
                f_att2 = (f_att2 -f_att2.min()) / (f_att2.max() - f_att2.min() + 1e-8)
                f_att2 = 255 * f_att2
                filename = os.path.join(self.config.test_folder, name[:-4] + 'f_att2_rgbonly.png')
                cv2.imwrite(filename, f_att2)
                corr_rgb2d2=(torch.sum(corr_rgb2d2,1)/corr_rgb2d2.shape[1]).unsqueeze(0)
                #corr_rgb2d2 = corr_rgb2d2[0].clone()
                corr_rgb2d2 = F.interpolate(corr_rgb2d2, tuple(im_size), mode='bilinear', align_corners=True)
                corr_rgb2d2 = corr_rgb2d2.sigmoid().data.cpu().numpy().squeeze()
                corr_rgb2d2 = (corr_rgb2d2 -corr_rgb2d2.min()) / (corr_rgb2d2.max() - corr_rgb2d2.min() + 1e-8)
                corr_rgb2d2 = 255 * corr_rgb2d2
                filename = os.path.join(self.config.test_folder, name[:-4] + 'corr_rgb2d2_rgbonly.png')
                cv2.imwrite(filename, corr_rgb2d2)
                corr_rgb2d3=(torch.sum(corr_rgb2d3,1)/corr_rgb2d3.shape[1]).unsqueeze(0)
                #corr_rgb2d3 = corr_rgb2d3[0].clone()
                corr_rgb2d3 = F.interpolate(corr_rgb2d3, tuple(im_size), mode='bilinear', align_corners=True)
                corr_rgb2d3 = corr_rgb2d3.sigmoid().data.cpu().numpy().squeeze()
                corr_rgb2d3 = (corr_rgb2d3 -corr_rgb2d3.min()) / (corr_rgb2d3.max() - corr_rgb2d3.min() + 1e-8)
                corr_rgb2d3 = 255 * corr_rgb2d3
                filename = os.path.join(self.config.test_folder, name[:-4] + 'corr_rgb2d3_rgbonly.png')
                cv2.imwrite(filename, corr_rgb2d3)
                corr_d2rgb2=(torch.sum(corr_d2rgb2,1)/corr_d2rgb2.shape[1]).unsqueeze(0)
                #corr_d2rgb2 = corr_d2rgb2[0].clone()
                corr_d2rgb2 = F.interpolate(corr_d2rgb2, tuple(im_size), mode='bilinear', align_corners=True)
                corr_d2rgb2 = corr_d2rgb2.sigmoid().data.cpu().numpy().squeeze()
                corr_d2rgb2 = (corr_d2rgb2 -corr_d2rgb2.min()) / (corr_d2rgb2.max() - corr_d2rgb2.min() + 1e-8)
                corr_d2rgb2 = 255 * corr_d2rgb2
                filename = os.path.join(self.config.test_folder, name[:-4] + 'corr_d2rgb2_rgbonly.png')
                cv2.imwrite(filename, corr_d2rgb2)
                corr_d2rgb3=(torch.sum(corr_d2rgb3,1)/corr_d2rgb3.shape[1]).unsqueeze(0)
                #corr_d2rgb3 = corr_d2rgb3[0].clone()
                corr_d2rgb3 = F.interpolate(corr_d2rgb3, tuple(im_size), mode='bilinear', align_corners=True)
                corr_d2rgb3 = corr_d2rgb3.sigmoid().data.cpu().numpy().squeeze()
                corr_d2rgb3 = (corr_d2rgb3 -corr_d2rgb3.min()) / (corr_d2rgb3.max() - corr_d2rgb3.min() + 1e-8)
                corr_d2rgb3 = 255 * corr_d2rgb3
                filename = os.path.join(self.config.test_folder, name[:-4] + 'corr_d2rgb3_rgbonly.png')
                cv2.imwrite(filename, corr_d2rgb3)

                coarse_sal_rgb=(torch.sum(coarse_sal_rgb,1)/coarse_sal_rgb.shape[1]).unsqueeze(0)
                #coarse_sal_rgb = coarse_sal_rgb[0].clone()
                coarse_sal_rgb = F.interpolate(coarse_sal_rgb, tuple(im_size), mode='bilinear', align_corners=True)
                coarse_sal_rgb = coarse_sal_rgb.sigmoid().data.cpu().numpy().squeeze()
                coarse_sal_rgb = (coarse_sal_rgb -coarse_sal_rgb.min()) / (coarse_sal_rgb.max() - coarse_sal_rgb.min() + 1e-8)
                coarse_sal_rgb = 255 * coarse_sal_rgb
                filename = os.path.join(self.config.test_folder, name[:-4] + 'coarse_sal_rgb_rgbonly.png')
                cv2.imwrite(filename, coarse_sal_rgb)
                coarse_sal_depth=(torch.sum(coarse_sal_depth,1)/coarse_sal_depth.shape[1]).unsqueeze(0)
                #coarse_sal_depth = coarse_sal_depth[0].clone()
                coarse_sal_depth = F.interpolate(coarse_sal_depth, tuple(im_size), mode='bilinear', align_corners=True)
                coarse_sal_depth = coarse_sal_depth.sigmoid().data.cpu().numpy().squeeze()
                coarse_sal_depth = (coarse_sal_depth -coarse_sal_depth.min()) / (coarse_sal_depth.max() - coarse_sal_depth.min() + 1e-8)
                coarse_sal_depth = 255 * coarse_sal_depth
                filename = os.path.join(self.config.test_folder, name[:-4] + 'coarse_sal_depth_rgbonly.png')
                cv2.imwrite(filename, coarse_sal_depth)
                e_rgbd0=(torch.sum(e_rgbd0,1)/e_rgbd0.shape[1]).unsqueeze(0)
                #e_rgbd0 = e_rgbd0[0].clone()
                e_rgbd0 = F.interpolate(e_rgbd0, tuple(im_size), mode='bilinear', align_corners=True)
                e_rgbd0 = e_rgbd0.sigmoid().data.cpu().numpy().squeeze()
                e_rgbd0 = (e_rgbd0 -e_rgbd0.min()) / (e_rgbd0.max() - e_rgbd0.min() + 1e-8)
                e_rgbd0 = 255 * e_rgbd0
                filename = os.path.join(self.config.test_folder, name[:-4] + 'e_rgbd0_rgbonly.png')
                cv2.imwrite(filename, e_rgbd0)
                e_rgbd1=(torch.sum(e_rgbd1,1)/e_rgbd1.shape[1]).unsqueeze(0)
                #e_rgbd1 = e_rgbd1[0].clone()
                e_rgbd1 = F.interpolate(e_rgbd1, tuple(im_size), mode='bilinear', align_corners=True)
                e_rgbd1 = e_rgbd1.sigmoid().data.cpu().numpy().squeeze()
                e_rgbd1 = (e_rgbd1 -e_rgbd1.min()) / (e_rgbd1.max() - e_rgbd1.min() + 1e-8)
                e_rgbd1 = 255 * e_rgbd1
                filename = os.path.join(self.config.test_folder, name[:-4] + 'e_rgbd1_rgbonly.png')
                cv2.imwrite(filename, e_rgbd1)
                e_rgbd2=(torch.sum(e_rgbd2,1)/e_rgbd2.shape[1]).unsqueeze(0)
                #e_rgbd2 = e_rgbd2[0].clone()
                e_rgbd2 = F.interpolate(e_rgbd2, tuple(im_size), mode='bilinear', align_corners=True)
                e_rgbd2 = e_rgbd2.sigmoid().data.cpu().numpy().squeeze()
                e_rgbd2 = (e_rgbd2 -e_rgbd2.min()) / (e_rgbd2.max() - e_rgbd2.min() + 1e-8)
                e_rgbd2 = 255 * e_rgbd2
                filename = os.path.join(self.config.test_folder, name[:-4] + 'e_rgbd2_rgbonly.png')
                cv2.imwrite(filename, e_rgbd2)
                rgb_1=(torch.sum(rgb_1,1)/rgb_1.shape[1]).unsqueeze(0)
                #rgb_1 = rgb_1[0].clone()
                rgb_1 = F.interpolate(rgb_1, tuple(im_size), mode='bilinear', align_corners=True)
                rgb_1 = rgb_1.sigmoid().data.cpu().numpy().squeeze()
                rgb_1 = (rgb_1 -rgb_1.min()) / (rgb_1.max() - rgb_1.min() + 1e-8)
                rgb_1 = 255 * rgb_1
                filename = os.path.join(self.config.test_folder, name[:-4] + 'rgb_1_rgbonly.png')
                cv2.imwrite(filename, rgb_1)
                rgb_2=(torch.sum(rgb_2,1)/rgb_2.shape[1]).unsqueeze(0)
                #rgb_2 = rgb_2[0].clone()
                rgb_2 = F.interpolate(rgb_2, tuple(im_size), mode='bilinear', align_corners=True)
                rgb_2 = rgb_2.sigmoid().data.cpu().numpy().squeeze()
                rgb_2 = (rgb_2 -rgb_2.min()) / (rgb_2.max() - rgb_2.min() + 1e-8)
                rgb_2 = 255 * rgb_2
                filename = os.path.join(self.config.test_folder, name[:-4] + 'rgb_2_rgbonly.png')
                cv2.imwrite(filename, rgb_2)
                rgb_3=(torch.sum(rgb_3,1)/rgb_3.shape[1]).unsqueeze(0)
                #rgb_3 = rgb_3[0].clone()
                rgb_3 = F.interpolate(rgb_3, tuple(im_size), mode='bilinear', align_corners=True)
                rgb_3 = rgb_3.sigmoid().data.cpu().numpy().squeeze()
                rgb_3 = (rgb_3 -rgb_3.min()) / (rgb_3.max() - rgb_3.min() + 1e-8)
                rgb_3 = 255 * rgb_3
                filename = os.path.join(self.config.test_folder, name[:-4] + 'rgb_3_rgbonly.png')
                cv2.imwrite(filename, rgb_3)
                rgb_4=(torch.sum(rgb_4,1)/rgb_4.shape[1]).unsqueeze(0)
                #rgb_4 = rgb_4[0].clone()
                rgb_4 = F.interpolate(rgb_4, tuple(im_size), mode='bilinear', align_corners=True)
                rgb_4 = rgb_4.sigmoid().data.cpu().numpy().squeeze()
                rgb_4 = (rgb_4 -rgb_4.min()) / (rgb_4.max() - rgb_4.min() + 1e-8)
                rgb_4 = 255 * rgb_4
                filename = os.path.join(self.config.test_folder, name[:-4] + 'rgb_4_rgbonly.png')
                cv2.imwrite(filename, rgb_4)
                rgb_5=(torch.sum(rgb_5,1)/rgb_5.shape[1]).unsqueeze(0)
                #rgb_5 = rgb_5[0].clone()
                rgb_5 = F.interpolate(rgb_5, tuple(im_size), mode='bilinear', align_corners=True)
                rgb_5 = rgb_5.sigmoid().data.cpu().numpy().squeeze()
                rgb_5 = (rgb_5 -rgb_5.min()) / (rgb_5.max() - rgb_5.min() + 1e-8)
                rgb_5 = 255 * rgb_5
                filename = os.path.join(self.config.test_folder, name[:-4] + 'rgb_5_rgbonly.png')
                cv2.imwrite(filename, rgb_5)
                depth_1=(torch.sum(depth_1,1)/depth_1.shape[1]).unsqueeze(0)
                #depth_1 = depth_1[0].clone()
                depth_1 = F.interpolate(depth_1, tuple(im_size), mode='bilinear', align_corners=True)
                depth_1 = depth_1.sigmoid().data.cpu().numpy().squeeze()
                depth_1 = (depth_1 -depth_1.min()) / (depth_1.max() - depth_1.min() + 1e-8)
                depth_1 = 255 * depth_1
                filename = os.path.join(self.config.test_folder, name[:-4] + 'depth_1_rgbonly.png')
                cv2.imwrite(filename, depth_1)
                depth_2=(torch.sum(depth_2,1)/depth_2.shape[1]).unsqueeze(0)
                #depth_2 = depth_2[0].clone()
                depth_2 = F.interpolate(depth_2, tuple(im_size), mode='bilinear', align_corners=True)
                depth_2 = depth_2.sigmoid().data.cpu().numpy().squeeze()
                depth_2 = (depth_2 -depth_2.min()) / (depth_2.max() - depth_2.min() + 1e-8)
                depth_2 = 255 * depth_2
                filename = os.path.join(self.config.test_folder, name[:-4] + 'depth_2_rgbonly.png')
                cv2.imwrite(filename, depth_2)
                depth_3=(torch.sum(depth_3,1)/depth_3.shape[1]).unsqueeze(0)
                #depth_3 = depth_3[0].clone()
                depth_3 = F.interpolate(depth_3, tuple(im_size), mode='bilinear', align_corners=True)
                depth_3 = depth_3.sigmoid().data.cpu().numpy().squeeze()
                depth_3 = (depth_3 -depth_3.min()) / (depth_3.max() - depth_3.min() + 1e-8)
                depth_3 = 255 * depth_3
                filename = os.path.join(self.config.test_folder, name[:-4] + 'depth_3_rgbonly.png')
                cv2.imwrite(filename, depth_3)
                depth_4=(torch.sum(depth_4,1)/depth_4.shape[1]).unsqueeze(0)
                #depth_4 = depth_4[0].clone()
                depth_4 = F.interpolate(depth_4, tuple(im_size), mode='bilinear', align_corners=True)
                depth_4 = depth_4.sigmoid().data.cpu().numpy().squeeze()
                depth_4 = (depth_4 -depth_4.min()) / (depth_4.max() - depth_4.min() + 1e-8)
                depth_4 = 255 * depth_4
                filename = os.path.join(self.config.test_folder, name[:-4] + 'depth_4_rgbonly.png')
                cv2.imwrite(filename, depth_4)
                depth_5=(torch.sum(depth_5,1)/depth_5.shape[1]).unsqueeze(0)
                #depth_5 = depth_5[0].clone()
                depth_5 = F.interpolate(depth_5, tuple(im_size), mode='bilinear', align_corners=True)
                depth_5 = depth_5.sigmoid().data.cpu().numpy().squeeze()
                depth_5 = (depth_5 -depth_5.min()) / (depth_5.max() - depth_5.min() + 1e-8)
                depth_5 = 255 * depth_5
                filename = os.path.join(self.config.test_folder, name[:-4] + 'depth_5_rgbonly.png')
                cv2.imwrite(filename, depth_5)
                rgbd_fusion_1=(torch.sum(rgbd_fusion_1,1)/rgbd_fusion_1.shape[1]).unsqueeze(0)
                #rgbd_fusion_1 = rgbd_fusion_1[0].clone()
                rgbd_fusion_1 = F.interpolate(rgbd_fusion_1, tuple(im_size), mode='bilinear', align_corners=True)
                rgbd_fusion_1 = rgbd_fusion_1.sigmoid().data.cpu().numpy().squeeze()
                rgbd_fusion_1 = (rgbd_fusion_1 -rgbd_fusion_1.min()) / (rgbd_fusion_1.max() - rgbd_fusion_1.min() + 1e-8)
                rgbd_fusion_1 = 255 * rgbd_fusion_1
                filename = os.path.join(self.config.test_folder, name[:-4] + 'rgbd_fusion_1_rgbonly.png')
                cv2.imwrite(filename, rgbd_fusion_1)
                rgbd_fusion_2=(torch.sum(rgbd_fusion_2,1)/rgbd_fusion_2.shape[1]).unsqueeze(0)
                #rgbd_fusion_2 = rgbd_fusion_2[0].clone()
                rgbd_fusion_2 = F.interpolate(rgbd_fusion_2, tuple(im_size), mode='bilinear', align_corners=True)
                rgbd_fusion_2 = rgbd_fusion_2.sigmoid().data.cpu().numpy().squeeze()
                rgbd_fusion_2 = (rgbd_fusion_2 -rgbd_fusion_2.min()) / (rgbd_fusion_2.max() - rgbd_fusion_2.min() + 1e-8)
                rgbd_fusion_2 = 255 * rgbd_fusion_2
                filename = os.path.join(self.config.test_folder, name[:-4] + 'rgbd_fusion_2_rgbonly.png')
                cv2.imwrite(filename, rgbd_fusion_2)
                rgbd_fusion_3=(torch.sum(rgbd_fusion_3,1)/rgbd_fusion_3.shape[1]).unsqueeze(0)
                #rgbd_fusion_3 = rgbd_fusion_3[0].clone()
                rgbd_fusion_3 = F.interpolate(rgbd_fusion_3, tuple(im_size), mode='bilinear', align_corners=True)
                rgbd_fusion_3 = rgbd_fusion_3.sigmoid().data.cpu().numpy().squeeze()
                rgbd_fusion_3 = (rgbd_fusion_3 -rgbd_fusion_3.min()) / (rgbd_fusion_3.max() - rgbd_fusion_3.min() + 1e-8)
                rgbd_fusion_3 = 255 * rgbd_fusion_3
                filename = os.path.join(self.config.test_folder, name[:-4] + 'rgbd_fusion_3_rgbonly.png')
                cv2.imwrite(filename, rgbd_fusion_3)
                rgbd_fusion_4=(torch.sum(rgbd_fusion_4,1)/rgbd_fusion_4.shape[1]).unsqueeze(0)
                #rgbd_fusion_4 = rgbd_fusion_4[0].clone()
                rgbd_fusion_4 = F.interpolate(rgbd_fusion_4, tuple(im_size), mode='bilinear', align_corners=True)
                rgbd_fusion_4 = rgbd_fusion_4.sigmoid().data.cpu().numpy().squeeze()
                rgbd_fusion_4 = (rgbd_fusion_4 -rgbd_fusion_4.min()) / (rgbd_fusion_4.max() - rgbd_fusion_4.min() + 1e-8)
                rgbd_fusion_4 = 255 * rgbd_fusion_4
                filename = os.path.join(self.config.test_folder, name[:-4] + 'rgbd_fusion_4_rgbonly.png')
                cv2.imwrite(filename, rgbd_fusion_4)
                rgbd_fusion_5=(torch.sum(rgbd_fusion_5,1)/rgbd_fusion_5.shape[1]).unsqueeze(0)
                #rgbd_fusion_5 = rgbd_fusion_5[0].clone()
                rgbd_fusion_5 = F.interpolate(rgbd_fusion_5, tuple(im_size), mode='bilinear', align_corners=True)
                rgbd_fusion_5 = rgbd_fusion_5.sigmoid().data.cpu().numpy().squeeze()
                rgbd_fusion_5 = (rgbd_fusion_5 -rgbd_fusion_5.min()) / (rgbd_fusion_5.max() - rgbd_fusion_5.min() + 1e-8)
                rgbd_fusion_5 = 255 * rgbd_fusion_5
                filename = os.path.join(self.config.test_folder, name[:-4] + 'rgbd_fusion_5_rgbonly.png')
                cv2.imwrite(filename, rgbd_fusion_5)'''

        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    
  
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        
        loss_vals=  []
        
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item=0
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_depth, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                    'sal_label'], data_batch['sal_edge']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label, sal_edge= sal_image.to(device), sal_depth.to(device), sal_label.to(device),sal_edge.to(device)

               
                self.optimizer.zero_grad()
                sal_label_coarse = F.interpolate(sal_label, size_coarse, mode='bilinear', align_corners=True)
                
                sal_final,coarse_sal_rgb,coarse_sal_depth,Att,e_rgbd0,e_rgbd1,e_rgbd2,rgb_1,rgb_2,\
rgb_3,rgb_4,rgb_5,depth_1,depth_2,depth_3,depth_4,depth_5,rgbd_fusion_1,rgbd_fusion_2,\
rgbd_fusion_3,rgbd_fusion_4,rgbd_fusion_5 = self.net(sal_image,sal_depth)
                
                sal_loss_coarse_rgb =  F.binary_cross_entropy_with_logits(coarse_sal_rgb, sal_label_coarse, reduction='sum')
                sal_loss_coarse_depth =  F.binary_cross_entropy_with_logits(coarse_sal_depth, sal_label_coarse, reduction='sum')
                sal_final_loss =  F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                '''edge_loss_rgbd0=F.smooth_l1_loss(sal_edge_rgbd0,sal_edge)
                edge_loss_rgbd1=F.smooth_l1_loss(sal_edge_rgbd1,sal_edge)
                edge_loss_rgbd2=F.smooth_l1_loss(sal_edge_rgbd2,sal_edge)'''
                
                sal_loss_fuse = sal_final_loss+sal_loss_coarse_rgb+sal_loss_coarse_depth
                sal_loss = sal_loss_fuse/ (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data
                r_sal_loss_item+=sal_loss.item() * sal_image.size(0)
                sal_loss.backward()
                self.optimizer.step()

                if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %0.4f  ||sal_final:%0.4f||  r:%0.4f||d:%0.4f' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss,sal_final_loss,sal_loss_coarse_rgb,sal_loss_coarse_depth ))
                    # print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', r_sal_loss / (self.show_every / self.iter_size),
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_loss_coarse_rgb training loss', sal_loss_coarse_rgb.data,
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_loss_coarse_depth training loss', sal_loss_coarse_depth.data,
                                      epoch * len(self.train_loader.dataset) + i)
                    

                    r_sal_loss = 0
                    res = coarse_sal_depth[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('coarse_sal_depth', torch.tensor(res), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)

                    res = coarse_sal_rgb[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('coarse_sal_rgb', torch.tensor(res), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)
                    
                    fsal = sal_final[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_final', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)

                    '''fsal = sal_low[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_low', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)
                    fsal = sal_high[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_high', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)
                    fsal = sal_med[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_med', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)'''
                    


            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            loss_vals.append(train_loss)
            
            print('Epoch:[%2d/%2d] | Train Loss : %.3f' % (epoch, self.config.epoch,train_loss))
            
        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)
        

