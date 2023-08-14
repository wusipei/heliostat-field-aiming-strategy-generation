# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 19:33:30 2022

@author: 86198
"""
import torch
from torchvision import transforms
from torch.autograd import Variable
import argparse
import os

import deal_with as dw

import numpy as np
import pandas as pd
import math


d = 3
out_d = 1

##^^^^^^#######pix2pix_train:data-driving training on given flux maps
check_dir = 'Two_Stage_model_on_given_flux_maps'
max_idx = 24

##^^^^^^#######E2E on given flux maps
check_dir2 = 'E2E_model_on_given_flux_maps'
max_idx2 = 1

G = torch.load(os.path.join(check_dir,'gen_model_'+str(max_idx)+'.pt'))
name = check_dir+'gen_model_'+str(max_idx)
G.eval()

G2 = torch.load(os.path.join(check_dir2,'gen_model_'+str(max_idx2)+'.pt'))
name = check_dir2+'gen_model_'+str(max_idx2)
G2.eval()

test_name = 'dataset_new_test_all'
save_dir = 'G_test_'+test_name+name
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

input_type = 'AE'

need_flux = True

train_data = dw.MyDataset(test_name+".txt", d, out_d, input_type, need_flux = need_flux)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=1,
                                                shuffle=False,
                                                drop_last=True)

all_data = []

G.cuda()

# Loss function
#gpu
BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()
L2_loss = torch.nn.MSELoss().cuda()

for epoch in range(1): 
    for i, (input1,input1_norm, target, w, ref_2D, txtname,az, ele) in enumerate(train_data_loader):   
        
        idx = dw.get_idx(az,ele)
        x_ = Variable(input1.cuda())
        x_norm = Variable(input1_norm.cuda())
        y_ = Variable(target.cuda())
        y_ = torch.unsqueeze(y_,1)
        
        
        gen_image = G(x_norm)
        
        gen_image2 = G2(x_)
        
        if i % 10 == 0:
            print(txtname[0])
        if i % 1== 0:
            im0 = input1[0].squeeze().cpu().detach().numpy()
            im1 = target[0].squeeze().cpu().detach().numpy()
            im2 = gen_image[0].squeeze().cpu().detach().numpy()
            im3 = gen_image2[0].squeeze().cpu().detach().numpy()
            epoch_name = 'epoch'+str(epoch)+'_batch'+str(i)+'_txtname'
            v0,v1,v2, GA_score, GA_s1, GA_s2, GA_flux, TwoStage_score, TwoStage_s1, TwoStage_s2, TwoStage_flux, AFD = \
                dw.calculate_one(int(az[0]),int(ele[0]),im0,im1,im2,epoch,pr=False,
                                 pandas=True)
            v0,v1,v2, GA_score, GA_s1, GA_s2, GA_flux, E2E_score, E2E_s1, E2E_s2, E2E_flux, AFD = \
                dw.calculate_one(int(az[0]),int(ele[0]),im0,im1,im3,epoch,pr=False,
                                 pandas=True)
            if i%10==0:
                dw.show_one_pred(int(az),int(ele),AFD,GA_flux,TwoStage_flux,E2E_flux,im0[[0,1,2],:],im1,im2,im3,epoch,save_dir = save_dir, add_name = epoch_name+str(txtname[0])[:-5])
                print(E2E_score)
            
            all_data.append([txtname[0], GA_score, GA_s1, GA_s2, TwoStage_score, TwoStage_s1, TwoStage_s2, E2E_score, E2E_s1, E2E_s2])
            

df = pd.DataFrame(all_data, columns=['txtname', 'GA_S_obj','GA_S_power','GA_S_exceed', 'TwoStage_S_obj','TwoStage_S_power','TwoStage_S_exceed', 'E2E_S_obj','E2E_S_power','E2E_S_exceed']) 

df.to_excel('test_result/'+test_name+name+'.xlsx')





