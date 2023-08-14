# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:39:13 2021

@author: 86198
"""
import math
import numpy as np
import pandas as pd
import torch 
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
from torchvision import transforms
import torchvision
from torchvision.datasets import mnist # 获取数据集
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import os
import imageio

def norm(AE_1):
    return (AE_1-AE_1.min())/(AE_1.max()-AE_1.min())

def max_norm(AE_1):
    return (AE_1/AE_1.max())

#######################################################################
#produce_code.py中的函数
#######################################################################
#math.modf=>(小数部分,整数部分)
def get_aimpoint_str(aimpoint):
    decimal = str(round(math.modf(aimpoint)[0]*100))
    integer = str(round(math.modf(aimpoint)[1]))
    aimpoint_str = integer + '_' + decimal
    return aimpoint_str

def get_one_flux(filename,dirname = '', realflux = False, realsolution_raw = False, realsolution_smooth = False, getzipflux = False):
    rootpath = 'flux'
    #选取实际无遮挡情况下的最佳光斑作为理想光斑
    if realflux:
        rootpath = 'realflux'
    elif realsolution_raw:
        rootpath = 'realsolution_raw'
    elif realsolution_smooth:
        rootpath = 'realsolution_smooth'
    elif getzipflux:
        rootpath = 'flux_40'
    filepath = os.path.join(rootpath,dirname,filename)
    with open(filepath,"r")as f:
        data = f.read()
    flux = [float(i) for i in data.split(',')]
    return flux

###########################################################################
#获取计算所需数据，包括：
#a.flux_all_all:(28,8,696,280)28种场景下的所有光斑
#b.AE_all:(28*4, 696)28种场景下自编码器压缩后的三位编码与光强
#c.ideal_flux_all:(28, 280)28种情形下的无遮挡优化光斑，用于计算分数score
#######################################################################
X_MIN = 171.25
DNA_SIZE = 3

flux_all_all = np.load('deal_with_data/flux_all_all.npy')
flux_all_centeral = flux_all_all[:,4].sum(axis=2).sum(axis=1)
flux_all_all_2D = np.load('deal_with_data/flux_all_all_2D.npy')
flux_all_all_40 = np.empty([28,8,696,40],dtype=np.float32)
for i in range(40):
    flux_all_all_40[:,:,:,i] = np.sum(flux_all_all[:,:,:,i*7:(i+1)*7],axis=3)
AE_all = np.load('deal_with_data/AE_all_log1020exp.npy')
I_all = np.load('deal_with_data/I_all.npy')
ideal_flux_all = np.load('deal_with_data/ideal_flux_all.npy')
AE_all_aim = np.load('deal_with_data/AE_all_aim.npy')
FLUX_MAX = 330000


def get_idx(az,ele):
    if az==360:
        az = 0
    idx = az//30*4 + (ele-10)//20
    return idx

def get_az_ele(idx):
    az = idx//4*30
    ele = idx%4*20+10
    return az,ele

def get_real_flux_all_all(flux_all_all):
    more_flux = np.zeros(28)
    for i in range(28):
        az,ele = get_az_ele(i)
        if abs(az-180) < 90 and ele < 90:
            irradiance = 1000
            more_flux[i] = irradiance * math.cos((az-180)/180*math.pi) * math.cos(ele/180*math.pi)
    real_flux_all_all = (flux_all_all.transpose(1,2,3,0)-more_flux).transpose(3,0,1,2)
    real_flux_all_all[real_flux_all_all<0]=0
    return real_flux_all_all

real_flux_all_all = get_real_flux_all_all(flux_all_all)

    
def get_ideal_flux_npa(az,ele):
    idx = get_idx(az,ele)
    return ideal_flux_all[idx].copy()

def get_flux_all_tmp(az,ele):
    idx = get_idx(az,ele)
    return flux_all_all[idx].copy()

#K=1.1
#ww = 10

K=1
ww = 1

def get_AFD(idx,ref):
    #基准AFD的计算方案一:(均值方案)
    #######true, random
    central_flux = flux_all_centeral[idx]
    #central_flux = flux_all_centeral
    #######
    az,ele = get_az_ele(idx)
    if abs(az-180) <= 90 and ele < 90:
        central_flux = central_flux- 1000* math.cos((az-180)/180*math.pi) * math.cos(ele/180*math.pi) \
            *280*( np.sum(ref)-1 )
    #最终所得central_flux表示 按照中心聚焦策略所得光斑 的 均值 计算出的基准AFD，
    #lambds考虑了不同阴影大小的影响。
    #其乘上1.3左右即为一个具有约束的AFD设定值，系数越大，AFD约束越小;
    if abs(az-180) <= 90 and ele < 90:
        more = 1000* math.cos((az-180)/180*math.pi) * math.cos(ele/180*math.pi) *280*( np.sum(ref)-1 )
    else:
        more = 0
    ######
    real_lambd = (np.sum(I_all[idx]*ref) - more) / (np.sum(I_all[idx]) - more)
    #print(I_all.shape,ref.shape)
    #real_lambd = (np.sum(I_all*ref) - more) / (np.sum(I_all) - more)
    ######
    central_flux = central_flux * real_lambd
    AFD = central_flux * K / 280
    return AFD


def get_one_fitness_AFD(AFD,position_array,reflectivity_npa,idx, needflux=False,txtname=''):
    grids=280
    ZEROS = 696-np.sum(reflectivity_npa)
    NONZEROS_idx = np.where(reflectivity_npa==1)
    aimpoint_array = np.around((position_array[NONZEROS_idx]-X_MIN)/20*(2**DNA_SIZE)).astype(int)
    ######true, random,共需要改三处
    flux_together = flux_all_all[idx,aimpoint_array,NONZEROS_idx,:].sum(axis=0).sum(axis=0)
    ######
    
    ##########################
    #and表示既要太阳从后方输入也要太阳并非垂直入射时才需要减去多余光斑
    ##########################
    #设置初值
    more_flux = 0
    az,ele = get_az_ele(idx)
    if abs(az-180) <= 90 and ele < 90:
        irradiance = 1000
        more_flux = irradiance * math.cos((az-180)/180*math.pi) * math.cos(ele/180*math.pi) * (695-ZEROS)
    flux_together = flux_together - more_flux
    score = -flux_together.mean()/AFD
    score1 = score
    score2 = 0
    exceedAFD = flux_together[flux_together>AFD]
    if len(exceedAFD)>0:
        score2 = (exceedAFD-AFD).sum()/280/AFD
        score += ww*score2
 
    if needflux==False:
        uniformity = round(float(flux_together.std()/flux_together.mean()),6)
        power = flux_together.sum()
        return score, score1,score2,uniformity, power, flux_together
    else:
        
        print('uniformity:',flux_together.std()/flux_together.mean())
        print('score1:',-flux_together.mean()/AFD)
        exceedAFD = flux_together[flux_together>AFD]
        if len(exceedAFD)>0:
            print('score2:',(exceedAFD-AFD).mean()/AFD)        
        
        f = open('print\\'+txtname,'w')
        f.write('AFD:'+str(int(AFD))+'\n')
        uniformity = round(float(flux_together.std()/flux_together.mean()),6)
        f.write('uniformity:'+str(uniformity)+'\n')
        score1 = round(float(-flux_together.mean()/AFD),6)
        f.write('score1:'+str(score1)+'\n')
        exceedAFD = flux_together[flux_together>AFD]
        if len(exceedAFD)>0:
            score2 = round(float((exceedAFD-AFD).mean()/AFD),8)
            f.write('score2:'+str(score2)+'\n')
        f.close()
        
        return score,flux_together



#具体改变位置并计算适应度
def get_one_fitness(az,ele,position_array,reflectivity_npa):
    ideal_flux_npa = get_ideal_flux_npa(az,ele)
    flux_all_tmp = get_flux_all_tmp(az,ele)
    grids=280
    flux_together = np.zeros(grids)
    #统计反射率为0的个数，以消除多减掉的直射光斑
    ZEROS = 0
    for i in range(len(position_array)):
        if reflectivity_npa[i] == 0:
            ZEROS += 1
            continue
        #aimpoint_num = round((position_array[i]-X_MIN)/20*(2**DNA_SIZE))
        aimpoint_num = np.around((position_array[i]-X_MIN)/20*(2**DNA_SIZE)).astype(int)
        #aimpoint_num = int(aimpoint_num)
        flux_tmp = flux_all_tmp[aimpoint_num][i]
        flux_together += flux_tmp * reflectivity_npa[i]
    #print(flux_together)
    ##########################
    #and表示既要太阳从后方输入也要太阳并非垂直入射时才需要减去多余光斑
    ##########################
    #设置初值
    more_flux = 0
    if abs(az-180) <= 90 and ele < 90:
        irradiance = 1000
        more_flux = irradiance * math.cos((az-180)/180*math.pi) * math.cos(ele/180*math.pi) * (695-ZEROS)
    flux_together = flux_together - more_flux
    score = -np.sum((flux_together-ideal_flux_npa)**2)

    gridarea = 20*3.5/grids
    power = np.sum(flux_together)*gridarea
    averageflux = np.mean(flux_together)
    std = np.std(flux_together)
    uniformity = std/averageflux
    return score,power,uniformity,flux_together
############################################################

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def to_L_one(image_array):
    image_array*=255
    im = Image.fromarray(image_array)
    im = im.convert('L')
    return im

def show_one_pred(az,ele,AFD, GA_flux, DL_flux,DL_flux2,ref,solution,pred,pred2,i,save_dir = 'gray',add_name = ""):
    if len(ref.shape)==2:
        d=1
        
    elif len(ref.shape)==3:
        #(2,70,17) (3,70,17)
        d=ref.shape[0]
    matplotlib.use('Agg')
    #plt.figure(figsize=(20,10))
    nums=7
    plt.figure(figsize=(5*1.2 * nums, 5*2.36 * 1))
    #添加总标题
    #print(AFD,str(round(AFD/1000,3)))
    plt.suptitle('AFD:'+str(round(AFD/1000,3))+' az:'+str(az)+' ele:'+str(ele), size=20,x=0.5,y=0.995)
    ref = ref.transpose(1,2,0)
    """
    plt.subplot(1,nums,1)
    plt.title('input')
    ref = ref.transpose(1,2,0)
    plt.imshow(ref/ref.max(), cmap = plt.cm.jet)
    """
    size=20
    #'family' : 'serif',
    font = {
        'weight' : 'normal',
        'size'   : 20,
        }
    
    #
    plt.subplot(1,nums,2)
    plt.title('GA solution', size=size)
    plt.imshow(solution, cmap='gray')
    cb = plt.colorbar()
    cb.set_label(label="Relative aiming height",fontdict=font)
    plt.xlabel('Column number', size=size)
    plt.ylabel('Circle number', size=size)
    
    
    
    #3d, output
    ##
    plt.subplot(1,nums,3)
    #plt.title('Two Stage model solution', size=size)
    plt.title('End2end solution', size=size)
    ref_true = to_2D(to_reflectivity(ref[:,:,0]))
    pred[ref_true==0]=0
    plt.imshow(pred, cmap='gray')
    cb = plt.colorbar()
    cb.set_label(label="Relative aiming height",fontdict=font)
    plt.xlabel('Column number', size=size)
    plt.ylabel('Circle number', size=size)
    
    plt.subplot(1,nums,1)
    plt.title('Cloud shadowing', size=size)
    plt.imshow(ref_true, cmap='gray')
    plt.xlabel('Column number', size=size)
    plt.ylabel('Circle number', size=size)
    
    GA_flux = GA_flux/1000
    DL_flux = DL_flux/1000
    DL_flux2 = DL_flux2/1000
    low = min(GA_flux.min(),DL_flux.min(),DL_flux2.min())
    high = max(GA_flux.max(),DL_flux.max(),DL_flux2.max())
    a=np.array(GA_flux)
    x = a.shape[0]
    y = a.shape[1]
    extent = 0, y*.5, 0, x*.5

    plt.subplot(1,nums,5)
    plt.title('GA flux', size=size)
    plt.imshow(GA_flux, cmap=plt.cm.jet, interpolation='bicubic',vmin=low,vmax=high,extent=extent)
    cb = plt.colorbar()
    cb.set_label(label='Energy flux density of one panel [kW/m$^2$]',fontdict=font)
    #周向 Circumferential distance
    plt.xlabel('Circumferential distance [m]', size=size)
    #轴向距离 axial distance
    plt.ylabel('Axial distance [m]', size=size)
    
    plt.subplot(1,nums,6)
    #plt.title('Two Stage model flux', size=size)
    plt.title('End2end model flux', size=size)
    plt.imshow(DL_flux, cmap=plt.cm.jet, interpolation='bicubic',vmin=low,vmax=high,extent=extent)
    cb = plt.colorbar()
    cb.set_label(label='Energy flux density of one panel [kW/m$^2$]',fontdict=font)
    #周向 Circumferential distance
    plt.xlabel('Circumferential distance [m]', size=size)
    #轴向距离 axial distance
    plt.ylabel('Axial distance [m]', size=size)
    
    plt.subplot(1,nums,7)
    #plt.title('End2end model flux', size=size)
    plt.title('Robust end2end model flux', size=size)
    plt.imshow(DL_flux2, cmap=plt.cm.jet, interpolation='bicubic',vmin=low,vmax=high,extent=extent)
    cb = plt.colorbar()
    cb.set_label(label='Energy flux density of one panel [kW/m$^2$]',fontdict=font)
    #周向 Circumferential distance
    plt.xlabel('Circumferential distance [m]', size=size)
    #轴向距离 axial distance
    plt.ylabel('Axial distance [m]', size=size)
    
    plt.subplot(1,nums,4)
    #plt.title('End2end model solution', size=size)
    plt.title('Robust end2end model solution', size=size)
    pred2[ref_true==0]=0
    plt.imshow(pred2, cmap='gray')
    cb = plt.colorbar()
    cb.set_label(label="Relative aiming height",fontdict=font)
    plt.xlabel('Column number', size=size)
    plt.ylabel('Circle number', size=size)
    
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.)
    
    if add_name == "":
        figname = str(i+1)++'.png'
    else:
        figname = add_name+'.png'
    savepath = os.path.join(save_dir,figname)
    plt.savefig(savepath)
   


def to_reflectivity(im_1d):
    v0 = to_vector(im_1d)
    reflectivity = np.empty([696],dtype=np.float32)
    for i in range(696):
        #惨痛教训，要记得查看：
        #pix2pix,regression
        if abs(v0[i])>10e-5:
            reflectivity[i]=1
        else:
            reflectivity[i]=0
    return reflectivity

#返回aim, 0-7
def find_similar_aim(code1, mirror, idx):
    dis = np.array([np.sum((AE_all_aim[i, idx*4+1:(idx+1)*4, mirror]-code1)**2) 
           for i in range(8)])
    return np.argmin(dis)

#code1 or code2(3,70,17), return a 2D numpy array (70,17)
#idx is 0-27, az and ele
def code2solution(code,idx):
    code_vector = np.empty([3,696],dtype=np.float32)
    for i in range(3):
        code_vector[i] = to_vector(code[i])
    
    solution_vector = np.empty([696],dtype=np.float32)
    for j in range(696):
        solution_vector[j] = find_similar_aim(code_vector[:,j], j, idx)
    return to_2D(((solution_vector*2.5)+1.25)/20)


#solution is (696), return code (3,696)
def solution2code(solution,idx):
    code_vector = np.empty([3,696],dtype=np.float32)
    for j in range(696):
        code_vector[:,j] = AE_all_aim[int(solution[j]), idx*4+1:(idx+1)*4, j]
    return code_vector


def calculate_one(az,ele,im0,im1,im2,i,pr=True,pandas=False):
    if az==360:
        az=0
    if pr==True:
        print('test'+str(i+1)+'\n')

    v0 = to_reflectivity(im1)
    v1 = to_vector(im1*20+170)
    v2 = to_vector(im2*20+170)
    
    idx = get_idx(az, ele)
    AFD = get_AFD(idx,v0)
    
    bestscore, bestscore1,bestscore2,bestuniformity, bestpower, GA_flux = get_one_fitness_AFD(AFD,v1,v0,idx)
    v2 = standard(v2.copy())
    score, score1,score2, uniformity, power, DL_flux = get_one_fitness_AFD(AFD,v2,v0,idx)

    if pr==True:
        print("GA\t"+"%e\t"%bestscore+"%e\t"%bestpower+str(round(bestuniformity,6))+'\n')
        print("pix2pix\t"+"%e\t"%score+"%e\t"%power+str(round(uniformity,6))+'\n')
        print(str(more)+" percent less\n")
        print(str(more2)+" percent more\n")
    
    if pandas==True:
        return v0,v1,v2,bestscore,bestscore1,bestscore2,GA_flux.reshape(40,7),\
            score,score1,score2,DL_flux.reshape(40,7),\
                    AFD
    else:
        return GA_flux.reshape(40,7),DL_flux.reshape(40,7)


def standard(v2):
    for i in range(len(v2)):
        label = round((v2[i]-171.25)/2.5)
        if label>=8:
            label=7
        elif label<0:
            label=0
        v2[i] = 171.25 + label*2.5
    return v2


def get_h_num_in_one_r():
    h_num_list = []
    r0 = 1500
    rGap = 20
    for r in range(1,71):
        rTmp = r0 - ( r-1 ) * rGap
        if r%2:
            l0 = 10
        else:
            l0 = 0
        h_num_tmp = len(range(math.floor(-3.1415927/16*rTmp/40),
                       math.ceil(3.1415927/16*rTmp/40)+1))
        h_num_list.append(h_num_tmp)
    return np.array(h_num_list)

h_num_npa = get_h_num_in_one_r()
#70*17==》1*696

def to_vector(two_dimension):
    used = 0
    width = np.max(h_num_npa)
    vector = np.array([],dtype=np.float32)
    for i in range(len(h_num_npa)):
        a = h_num_npa[i]
        tmp = two_dimension[i]
        core = tmp

        pad = int((width-a)/2)
        vector = np.append(vector,core[pad:width-pad])
    return vector


#1*696===》17*70
def to_2D(one_dimension_npa):
    used = 0
    width = np.max(h_num_npa)
    two_dimension = np.empty([0,width], dtype=np.float32)
    for i in range(len(h_num_npa)):
        a = h_num_npa[i]
        tmp_npa = np.zeros(width)
        pad = int((width-a)/2)
        tmp_npa[pad:width-pad]= one_dimension_npa[used:used+a]
        two_dimension = np.append(two_dimension,[tmp_npa], axis=0)
        used+=a
    return two_dimension.astype(dtype=np.float32)

def to_2D_padding(one_dimension_npa, az, ele):
    #h_num_npa = get_h_num_in_one_r()
    used = 0
    width = np.max(h_num_npa)
    #two_dimension = np.empty([0,width*2], dtype=np.float32)
    two_dimension = np.empty([0,width], dtype=np.float32)
    for i in range(len(h_num_npa)):

        a = h_num_npa[i]
        pad = int((width-a)/2)
        tmp_npa = np.zeros(width)
        tmp_npa[:pad-1] = (az/180+0.1)
        tmp_npa[width-pad+1:] = (ele/90+0.1)
        tmp_npa[pad:width-pad]= one_dimension_npa[used:used+a]

        two_dimension = np.append(two_dimension,[tmp_npa], axis=0)
        #two_dimension = np.append(two_dimension,[add_npa], axis=0)
        
        #strip_npa = np.zeros([width*2], dtype=np.float32)
        #two_dimension = np.append(two_dimension,[strip_npa], axis=0)
        used+=a
    return two_dimension.astype(dtype=np.float32)

#划分测试集
#######################################################################
def Is_test(txtname,scale):
    testnum = [i for i in range(7,scale,10)]
    for i in range(len(testnum)):
        if txtname.find('test'+str(testnum[i])+'_')!=-1:
            return True
    return False

def get_scale(filename):
    f0 = open(filename,"r")
    scale = len(f0.readlines())/6
    scale = int(scale)
    f0.close()
    print("filename:",filename,"scale:",scale)
    return scale

def split_dataset(filename,trainfilename,testfilename):
    scale = get_scale(filename)
    with open(filename,"r") as f:
        for i in range(scale):
            #line1
            txtname = f.readline()
            #if Is_test(txtname,scale):
            if i%40==1:
                f2 = open(testfilename,"a")
            else:
                f2 = open(trainfilename,"a")
            f2.write(txtname)
            for i in range(5):
                tmp = f.readline()
                f2.write(tmp)
    return


#此处ref_tmp是指单个solution的696数组
def replace_AE(ref_tmp,az,ele):
    #mid needs to choose the 32th point
    
    d = 4
    idx = get_idx(az,ele)
    AE_1 = AE_all[idx*4 : (idx+1)*4][:]
    AE_1_torch = torch.from_numpy(AE_1)
    AE_1_torch = AE_1_torch.view(-1,696)
    AE_1 = AE_1_torch.numpy()
    
    '''
    d = 3
    AE_1 = AE_all[32][:]
    AE_1 = norm(AE_1)
    '''
    #print(AE_1.shape)
    #归一化
    
    for j in range(d):
        AE_1[j,0:696] = norm(AE_1[j,0:696])
    
    AE_mirror = np.empty([d,696],dtype=np.float32)
    for i in range(696):
        if round(ref_tmp[i])==0:
            AE_mirror[:,i] = 0
        elif round(ref_tmp[i])==1:
            AE_mirror[:,i] = AE_1[:,i]
        else:
            print('ERROR AE_replace')
            quit()
    return AE_mirror
        

class MyDataset(Dataset):
    def __init__(self, filename,inputdim=1,outputdim=1,type11='b', maxnum=200000, need_flux=False):
        # 这个list存放所有图像的地址
        dsname = filename[:filename.find('.txt')]
        rootpath = os.path.join('dataset_np', dsname)

        self.ref = np.load(os.path.join('dataset_np',dsname,'ref.npy'))
        self.solution = np.load(os.path.join('dataset_np',dsname,'solution.npy'))
        self.az_all = np.load(os.path.join('dataset_np',dsname,'az_all.npy'))
        self.ele_all = np.load(os.path.join('dataset_np',dsname,'ele_all.npy'))
        self.az_all = self.az_all.astype(np.int32).squeeze()
        self.ele_al = self.ele_all.astype(np.int32).squeeze()
        #详细计算过程参看 calculate_flux_dataset.py
        if need_flux == True:
            pass
            #self.flux_all = np.load(os.path.join('dataset_np',dsname,'flux_all.npy'))
            #self.flux_all /= FLUX_MAX

        self.txtname_all = np.load(os.path.join('dataset_np',dsname,'txtname_all.npy'))   
        self.scale = len(self.az_all)
        
        if self.scale>maxnum:
            self.ref = self.ref[:maxnum]
            self.solution = self.solution[:maxnum]
            self.az_all = self.az_all[:maxnum]
            self.ele_all = self.ele_all[:maxnum]
            self.txtname_all = self.txtname_all[:maxnum]
            self.scale = maxnum
            
        self.inputdim=inputdim
        self.outputdim=outputdim
        self.type11 = type11
        self.need_flux = need_flux

        
        
        
    def __getitem__(self, index):
        # 读取图像数据并返回
        # 这里的open_image是读取图像函数，可以用PIL、opencv等库进行读取
        ref,solution,az,ele,txtname = self.ref[index],self.solution[index],self.az_all[index],self.ele_all[index],self.txtname_all[index]

        new_ref,new_ref_norm,new_solution,new_intensity = MyTransform(ref.copy(),solution.copy(),az,ele,
                                                         self.inputdim,self.outputdim,
                                                         needweight=True,
                                                         type11=self.type11,
                                                         need_flux=self.need_flux)
        ref_2D = np.empty([1,70,17],dtype=np.float32)
        ref_2D[0] = to_2D(ref)
        if self.need_flux==True:
            return new_ref,new_ref_norm, new_solution, new_intensity, ref_2D, txtname,az,ele
        else:
            return new_ref,new_solution,new_intensity,txtname,az,ele

    def __len__(self):
        # 返回图像的数量
        return self.scale



def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

def MyTransform(ref,solution,az,ele,inputdim=4,outputdim=1,needweight=False,type11 = 'a', need_flux=False):
    if type11=='raw':
        return ref,solution,[]

    if az==360:
        az=0 
    idx = get_idx(az,ele)
    zero_idx = np.where(ref==0)
    
    ################################################    
    #part1.transform input
    if inputdim==3 and type11=='AE':
        d = 4
        AE_1 = np.empty([d-1,696],dtype=np.float32)
        AE_1_norm = np.empty([d-1,696],dtype=np.float32)
        idx = int(idx[0])
        AE_1_tmp = AE_all[idx*(d)+1 : (idx+1)*(d)].copy()
        AE_1[0:d-1,:] = AE_1_tmp
        AE_1_norm[0:d-1,:] = AE_1_tmp
        #print(AE_1.shape)
        for j in range(d-1):
            AE_1_norm[j,0:696] = norm(AE_1[j,0:696])
        AE_1[:,zero_idx] = 0
        AE_1_norm[:,zero_idx] = 0
        new_ref = np.empty([d-1,70,17],dtype=np.float32)
        new_ref_norm = np.empty([d-1,70,17],dtype=np.float32)
        for k in range(d-1):
            new_ref[k] = to_2D(AE_1[k])
            new_ref_norm[k] = to_2D(AE_1_norm[k])
    ################################################    
    #part2.transform solution
    if outputdim==1:
        new_solution = (solution.copy()-170)/20
        new_solution[zero_idx] = 0
        new_solution = to_2D(new_solution)


    if needweight==True:
        #intensity_1 = np.empty([696],dtype=np.float32)
        intensity_1 = I_all[idx].copy()
        intensity_1[zero_idx] = 0
        intensity_1 = norm(intensity_1)
        #new_intensity = np.empty([1,70,17],dtype=np.float32)
        new_intensity = to_2D(intensity_1)
        return new_ref, new_ref_norm, new_solution, new_intensity
    else:
        return new_ref, new_solution




    