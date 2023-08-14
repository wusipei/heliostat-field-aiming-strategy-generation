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
import numpy as np
import pandas as pd
import math

import deal_with as dw

d = 3
out_d = 1

#w715: E2E on unseen flux maps
check_dir = 'E2E_model_on_unseen_flux_maps'
max_idx = 0
#######w716: Robust-E2E on unseen flux maps
check_dir2 = 'Robust_E2E_model_on_unseen_flux_maps'
max_idx2 = 0

G = torch.load(os.path.join(check_dir,'gen_model_'+str(max_idx)+'.pt'))
name = check_dir+'gen_model_'+str(max_idx)

#G.eval()
G.train()


G2 = torch.load(os.path.join(check_dir2,'gen_model_'+str(max_idx2)+'.pt'))
name = check_dir2+'gen_model_'+str(max_idx2)
G2.train()



train_name = 'dataset_new_test_all'


save_dir = 'G_test_'+train_name+name
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


test_name = 'dataset_new_all'
input_type = 'AE'
need_flux = True
train_data = dw.MyDataset(train_name+".txt", d, out_d, input_type, need_flux = need_flux)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=1,
                                                shuffle=False,
                                                drop_last=True)


G.cuda()

# Loss function
#gpu
BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()
L2_loss = torch.nn.MSELoss().cuda()

all_data = []

from sklearn.cluster import KMeans
#28个太阳角度，随机挑选20种作为训练集，不重复抽样：np.random.sample,剩下的8个用于测试
#np.random.choice(list(range(28)),20,replace = False)
IDXS=[ 0, 26, 25, 27,  4, 22,  7, 21, 10,  9, 19, 16, 15,  1, 17, 24, 11, 13,  8, 23]


for epoch in range(1): 
    #for i, (input1, target, w, az, ele) in enumerate(train_data_loader):
    for i, (input1,input1_norm, target, w, ref_2D, txtname,az, ele) in enumerate(train_data_loader):   
        
        idx = dw.get_idx(az,ele)
        if idx in IDXS:
            continue
        x_ = Variable(input1.cuda())
        y_ = Variable(target.cuda())
        y_ = torch.unsqueeze(y_,1)
        
        gen_image = G(x_)
        
        gen_image2 = G2(x_)
        
        if i % 10 == 0:
            print(txtname[0])
        if i % 1== 0:
            im0 = input1[0].squeeze().cpu().detach().numpy()
            im1 = target[0].squeeze().cpu().detach().numpy()
            im2 = gen_image[0].squeeze().cpu().detach().numpy()
            im3 = gen_image2[0].squeeze().cpu().detach().numpy()
            #output为2d时采用下面代码进行可视化
            #im1 = target[0,0].squeeze().cpu().detach().numpy()
            #im2 = gen_image[0,0].squeeze().cpu().detach().numpy()
            #more,GA_flux,DL_flux = dw.calculate_one(int(az[0]),int(ele[0]),im0,im1,im2,epoch,pr=True)
            epoch_name = 'epoch'+str(epoch)+'_batch'+str(i)+'_txtname'
            v0,v1,v2, GA_score, GA_s1, GA_s2, GA_flux, E2E_score, E2E_s1, E2E_s2, E2E_flux, AFD = \
                dw.calculate_one(int(az[0]),int(ele[0]),im0,im1,im2,epoch,pr=False,
                                 pandas=True)
            v0,v1,v2, GA_score, GA_s1, GA_s2, GA_flux, Robust_score, Robust_s1, Robust_s2, Robust_flux, AFD = \
                dw.calculate_one(int(az[0]),int(ele[0]),im0,im1,im3,epoch,pr=False,
                                 pandas=True)
            if i%3==0:
                dw.show_one_pred(int(az), int(ele), AFD, GA_flux, E2E_flux, Robust_flux,im0[[0,1,2],:],im1,im2,im3,epoch,save_dir = save_dir, add_name = epoch_name+str(txtname[0])[:-5])
                print(Robust_score)
            
            all_data.append([txtname[0], GA_score, GA_s1, GA_s2, E2E_score, E2E_s1, E2E_s2, Robust_score, Robust_s1, Robust_s2 ])
            

df = pd.DataFrame(all_data, columns=['txtname', 'GA_S_obj','GA_S_power','GA_S_exceed', 'E2E_S_obj','E2E_S_power','E2E_S_exceed', 'Robust_S_obj','Robust_S_power','Robust_S_exceed']) 

df.to_excel('test_result/'+train_name+name+'.xlsx')





