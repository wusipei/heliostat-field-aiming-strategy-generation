import torch
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
import argparse
import os
import numpy as np
import math
import re
import sys
from scipy import stats

from model import Generator, Discriminator, AE_conv
import deal_with as dw


torch.set_default_tensor_type(torch.FloatTensor)
def print_model(model, filepath):
    f = open(filepath,'w')
    f.write(str(model))
    f.close()
    return


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='', help='input dataset')
parser.add_argument('--batch_size', type=int, default=5, help='train batch size')
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=5000, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=2e-6, help='learning rate for generator, default=0.0002')
#parser.add_argument('--lrD', type=float, default=2e-6, help='lGearning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--w', type=int, default=716, help='model index')
parser.add_argument('--k', type=float, default=1, help='K_AFD')
parser.add_argument('--v', type=float, default=1, help='v_multi_obj')

#k越大，AFD越大，总能量越大，超出惩罚项越小，总损失函数越小
#原来的起始强度：-0.93，现在的起始强度-0.88
params = parser.parse_args()
print(params)

AE_model = AE_conv(inputdim=8)

#log
class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

#sys.stdout = Logger("D:\\PIX2PIX\\Log\\w"+str(params.w)+".txt") #这里我将Log输出到D盘

d = 3
out_d = 1
train_name = 'dataset_new_train_all'
input_type = 'AE'
need_flux = True

train_data = dw.MyDataset(train_name+".txt", d, out_d, input_type, need_flux = need_flux)

train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=params.batch_size,
                                                shuffle=False,
                                                drop_last=True)
flux_w = params.w
print(flux_w)

add_name = 'AEreal_'+input_type+('_need_flux' if need_flux==True else '')+\
    '_flux_w'+str(flux_w)+'_GArawObj'+'_in'+str(d)+'_out'+str(out_d)+'_'+train_name+'_scale'+str(train_data.scale)+'_ngf'+str(params.ngf)+'_ndf'+str(params.ndf)+'_20230526'

has_weight = False
if has_weight:
    add_name += 'has_weight'

save_dir = 'gray'+add_name
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
else:
    print('WRONG')
    
start_epoch = 0
check_dir = 'check_points_'+add_name
if not os.path.exists(check_dir):    
    os.mkdir(check_dir)
    G = Generator(d, params.ngf, out_d)
    D = Discriminator(d+out_d, params.ndf, out_d)
    G.normal_weight_init()
    D.normal_weight_init()

elif len(os.listdir(check_dir))==0:
    G = Generator(d, params.ngf, out_d)
    D = Discriminator(d+out_d, params.ndf, out_d)
    G.normal_weight_init()
    D.normal_weight_init()

else:
    model_list = os.listdir(check_dir)
    max_idx = max([int(re.findall(r'\d+', i)[0]) for i in model_list])
    start_epoch = max_idx+1
    G = torch.load(os.path.join(check_dir,'gen_model_'+str(max_idx)+'.pt'))

ANN = torch.nn.Sequential(torch.nn.Linear(1, 32),
                          torch.nn.Sigmoid(),
                          torch.nn.Linear(32, 64),
                          torch.nn.Sigmoid(),
                          torch.nn.Linear(64, 32),
                          torch.nn.Sigmoid(),
                          torch.nn.Linear(32, 8),
                          torch.nn.Sigmoid())

#OH = torch.load('onehot.pt')
OH = torch.load('onehot_sigmoid.pt')
OH.to(torch.float32)
OH.cpu()
OH.zero_grad()
OH.eval()


#gpu
G.cuda()

# Loss function
#gpu
BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()
L2_loss = torch.nn.MSELoss().cuda()


# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))

###########
G_avg_losses = []

G_avg_losses_test = []

step = 0
G_losses = []
L1_losses = []
flux_losses = []




flux_all_all_2D = Variable(torch.from_numpy(dw.flux_all_all_2D.transpose(0,4,2,3,1)))


#aa1.flux_all_all_1: (280, 70, 17, 8)某种角度下的所有光斑
#aa2.ref_2D: (70,17)
#被证明是正确的 GA和原来MyDataset类中计算的是相同的AFD
def get_AFD(flux_all_all_1, ref_2D, az, ele):
    
    flux_all_centeral = flux_all_all_1[:,:,:,4].sum()
    #I_all:(70,17),注意 flux_all_all_1 与scikit-opt-try.py中的flux_all_all的顺序不一样
    I_all = flux_all_all_1.sum(axis=0).mean(axis=2)

    #基准AFD的计算方案一:(均值方案)
    central_flux = flux_all_centeral
    if abs(az-180) <= 90 and ele < 90:
        central_flux = central_flux- 1000* math.cos((az-180)/180*math.pi) * math.cos(ele/180*math.pi) \
            *280*( (ref_2D).sum()-1 )

    #最终所得central_flux表示 按照中心聚焦策略所得光斑 的 均值 计算出的基准AFD，
    #lambds考虑了不同阴影大小的影响。
    #其乘上1.3左右即为一个具有约束的AFD设定值，系数越大，AFD约束越小;
    if abs(az-180) <= 90 and ele < 90:
        more = 1000* math.cos((az-180)/180*math.pi) * math.cos(ele/180*math.pi) *280*( (ref_2D).sum()-1 )
    else:
        more = 0
    real_lambd = ((I_all*ref_2D).sum() - more) / ((I_all).sum() - more)
    #print(real_lambd, (I_all).sum() , more)
    central_flux = central_flux * real_lambd

    AFD = central_flux * params.k / 280
    return AFD



##from AE_train_new_conv_try import AE_conv
#log1020exp, 70,40,70
model = torch.load( 'AEmodels/AE_conv_8_3_real_flux_all_all_leakeyRELU_log1020exp.pth')

#取出解码器
decoder = nn.Sequential(*list(model.children())[5:10])
decoder.cuda().eval()

#inputcode:(3, 70, 17)
#outpurflux:(280, 70, 17, 8)
def code2flux(inputcode):
    ip = torch.where(inputcode!=0)
    
    #(10,8,40,7)
    b = torch.zeros(70,17,8,280)
    #(3,1032)->(1032,3,1,1)->(1032,8,40,7)
    code = inputcode[:,ip[1],ip[2]].permute(1,0).unsqueeze(2).unsqueeze(3)

    bnj_exp = torch.exp(decoder(code.cuda()).cpu().view(-1,8,280))-1.1

    b[ip[1],ip[2],:,:] = bnj_exp
    flux_all_all_decoder = b.permute(3,0,1,2)
    #原因未知，该点总会出现某些极大的强度，故进行平滑滤波，这对神经网络的权值更新有着重要的作用
    flux_all_all_decoder[11] = (flux_all_all_decoder[10]+flux_all_all_decoder[12])/2
    return torch.relu(flux_all_all_decoder)

#aa1.flux_all_all_1: (280, 70, 17, 8)某种角度下的所有光斑
#aa2.ref_2D: (70,17)
#AE_conv_8_leakeyRELU_log_add_1_1模型是去除了直射误差后训练的，即不需要根据太阳角度减去重复仿真的部分
def get_AFD_no_direct(flux_all_all_1,ref_2D):
    
    flux_all_centeral = flux_all_all_1[:,:,:,4].sum()
    central_flux = flux_all_centeral
    AFD = central_flux * params.k / 280
    return AFD


for epoch in range(start_epoch, params.num_epochs): 
    #if epoch>start_epoch:
    #    f = open('Log/summary_'+str(params.w)+'.txt','a')
    #    f.write(str(float(flux_loss_sum/ij))+'\n')
    #    f.close()
    flux_loss_sum = 0
    ij = 0
    G.train()

    # training
    for i, ( input1, target, target_flux, w, ref_2D, txtname,az, ele, central_flux) in enumerate(train_data_loader):   

        x_ = Variable(input1.cuda())

        y_ = Variable(target.cuda())
        target_flux_ = Variable(target_flux.cuda())

        y_ = torch.unsqueeze(y_,1)

        
        gen_image = G(x_)
        

        a = gen_image.cpu().permute(0,2,3,1)
        solution_onehot = OH(a)

        gen_flux = []
        AFD3s = central_flux.detach()
        for ii in range(params.batch_size):
            az_tmp = int(az[ii])
            ele_tmp = int(ele[ii])
            idx = dw.get_idx(az[ii], ele[ii])
            #(280,70,17,8)*(70,17,8)=(280,70,17,8)
            #sum(axis=3)+squeeze:(280,70,17,8)==>(280,70,17)
            #(280,70,17)*(70,17)=(280,70,17)
            #sum(axis=[1,2]):(280,70,17)==>(280,)
            #reshape(40,7)
            ######
            #gen_flux_tmp = (flux_all_all_2D[idx]*solution_onehot[ii])
            #flux_all_all_2D_rand = torch.zeros((280,70,17,8))
            #flux_all_all_2D_rand[:,rand_idx_x,rand_idx_y,:] = flux_all_all_2D[idx,:,raw_idx_x,raw_idx_y,:]
            #gen_flux_tmp = (flux_all_all_2D_rand*solution_onehot[ii])
            ######
            
            #AFD2 = get_AFD(flux_all_all_2D[idx],ref_2D[ii],az_tmp,ele_tmp)
            
            
            """
            gen_flux_tmp = (flux_all_all_2D[idx]*solution_onehot[ii])
            gen_flux_tmp = gen_flux_tmp.sum(axis=3).squeeze()
            gen_flux_tmp = gen_flux_tmp*ref_2D[ii]
            gen_flux_tmp = gen_flux_tmp.sum(axis=[1,2])
            more_flux = 0
            if abs(az_tmp-180) <= 90 and ele_tmp < 90:
                irradiance = 1000
                more_flux = irradiance * math.cos((az_tmp-180)/180*math.pi) * math.cos(ele_tmp/180*math.pi) * (ref_2D[ii].sum()-1)
            gen_flux_tmp = (gen_flux_tmp-more_flux)/330000
            
            """
            #Robust end-to-end model
            #function code2flux transform random codes into corresponding flux maps
            flux_all_all_decoder = code2flux(input1[ii])
            AFD3 = get_AFD_no_direct(flux_all_all_decoder, ref_2D[ii])
            AFD3s[ii] = AFD3
            gen_flux_tmp = (flux_all_all_decoder*solution_onehot[ii])
            gen_flux_tmp = gen_flux_tmp.sum(axis=3).squeeze()
            more_flux = 0
            gen_flux_tmp = (gen_flux_tmp.sum(axis=[1,2])-more_flux)/330000
            
            gen_flux_tmp = gen_flux_tmp.reshape(40,7)
            gen_flux.append(gen_flux_tmp.unsqueeze(0))
        
        gen_flux = torch.cat(gen_flux, dim=0)
        ######
        gen_flux = gen_flux.cuda()
        ###
        #I_tmp = I_all[az][ele]/330000
        #uniform_target_flux_ = Variable((torch.zeros(D_fake_decision.size())*I_tmp).cuda())
        #uniform_target_flux_ = Variable((torch.zeros(D_fake_decision.size())).cuda()*target_flux_.mean())
        #flux_loss = params.lamb * L1_loss(gen_flux, target_flux_) / target_flux_.mean()
        #flux_loss = params.lamb*100 * L2_loss(gen_flux, target_flux_) #/ target_flux_.mean()#有误，一个batch有五个均值，而这个只除了一个均值
        ############
        #  GA OBJ  # 
        ############
        #flux_loss = params.lamb * L2_loss(gen_flux, target_flux_) #/ target_flux_.mean()#有误，一个batch有五个均值，而这个只除了一个均值
        
        
        #######################
        #  MAX UNDER AFD OBJ  # 
        #######################
        #k = 1.5
        #此前计算出GA算法目标下在大多数的角度截断系数为85%-95%
        #在某些ele=10,30,az=150,180下截断系数为50-65%
        #AFD方案一：基于均值，k=1.3
        #solution free training
        #AFD = central_flux * params.k / 280 / 330000
        #random pre-evalution
        AFD = AFD3s/330000
        #
        AFD = AFD.cuda().unsqueeze(1).unsqueeze(2)
        
        #损失函数的归一化，平衡不同角度之间的总能量差异
        #this loss function is the key point of end2end / robust end2end model 
        flux_loss = params.lamb * (-(gen_flux/AFD).mean() + params.v * ((torch.relu(gen_flux-AFD)/AFD).mean()))

        #G_loss = l1_loss
        G_loss = flux_loss
        
        flux_loss_sum += G_loss
        ij += 1
        
        OH.zero_grad()
    
        G.zero_grad()

        G_loss.backward()

        G_optimizer.step()

        if i % 10 == 0:
            print('Epoch [%d/%d], Step [%d/%d],\n\
                  flux_loss: %.4f'
              % (epoch+1, params.num_epochs, i+1, len(train_data_loader),\
                 flux_loss.data))
            print(-(gen_flux/AFD).mean().cpu().detach().numpy(),\
              (torch.relu(gen_flux-AFD)/AFD).mean().cpu().detach().numpy())
            print(str(txtname[0])[:-5])
            print(AFD)
                
            #f = open('Log/flux_loss_'+str(params.w)+'.txt','a')
            #f.write(str(float(flux_loss))+'\n')
            #f.close()

        if i % 10== 0:

            G.eval()
            gen_image = G(x_)
            G.train()
            GA_flux = target_flux[0].cpu().detach().numpy()
            DL_flux = gen_flux[0].cpu().detach().numpy()
            
            epoch_name = 'epoch'+str(epoch)+'_batch'+str(i)+'_txtname'

            central_power = float(AFD3s[0])*0.25*280
            power = DL_flux.sum()*0.25*330000
            uniformity = DL_flux.std()/DL_flux.mean()
            print("pix2pix\t"+"%e\t"%power+str(round(uniformity,6)))
            print('intercept(截断率)：',power/central_power,'\n')
            print()
            im0 = input1[0].squeeze().cpu().detach().numpy()
            im2 = gen_image[0].squeeze().cpu().detach().numpy()
            dw.show_one_pred_new(DL_flux, im0[[0,1,2],:], im2, i, save_dir = save_dir, add_name = epoch_name+str(txtname[0])[:-5])
            
        
        #step += 1
        if i % 300 == 100:
            torch.save(G,os.path.join(check_dir,'gen_model_{}.pt'.format(epoch)))
            print_model(G,os.path.join(check_dir,'gen_model_{}.txt'.format(epoch)))

    torch.save(G,os.path.join(check_dir,'gen_model_{}.pt'.format(epoch)))
    print_model(G,os.path.join(check_dir,'gen_model_{}.txt'.format(epoch)))
    




