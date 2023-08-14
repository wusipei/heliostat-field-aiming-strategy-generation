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
#import produce_code as pc
#import imageio

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

#a1.flux_all_all:(28,8,696,280)28种场景下的所有光斑
#np.save('deal_with_data/flux_all_all.npy', flux_all_all)
flux_all_all = np.load('deal_with_data/flux_all_all.npy')
flux_all_all_2D = np.load('deal_with_data/flux_all_all_2D.npy')
########add flux 40 (after zip)
#a2.flux_all_all_40:(28,8,696,40)28种场景下的所有光斑
flux_all_all_40 = np.empty([28,8,696,40],dtype=np.float32)
for i in range(40):
    flux_all_all_40[:,:,:,i] = np.sum(flux_all_all[:,:,:,i*7:(i+1)*7],axis=3)

flux_zip_8 = np.empty([28,696,40],dtype=np.float32)
for i in range(8):
    for j in range(5):
        pass
        #flux_zip_8[:,:,i*5+j] = np.sum(flux_all_all_40[:,i,:,j*8:(j+1)*8],axis=2)

#AE_all = np.load('deal_with_data/AE_all_real_sigmoid.npy')
#np.save( 'deal_with_data/AE_all_real_log.npy',AE_all)
#AE_all = np.load('deal_with_data/AE_all_real_log.npy')
AE_all = np.load('deal_with_data/AE_all_log1020exp.npy')


def get_az_ele(idx):
    az = idx//4*30
    ele = idx%4*20+10
    return az,ele

#28个太阳角度，随机挑选20种作为训练集，不重复抽样：np.random.sample,剩下的8个用于测试
#np.random.choice(list(range(28)),20,replace = False)
IDXS=[ 0, 26, 25, 27,  4, 22,  7, 21, 10,  9, 19, 16, 15,  1, 17, 24, 11, 13,  8, 23]
test_IDXS = [i for i in range(28) if i not in IDXS]

AE_data = np.zeros((20,696*3))
for i in range(20):
    idx = IDXS[i]
    for j in range(3):
        AE_data[i,j*696:(j+1)*696] = AE_all[idx*4+j+1]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(AE_data)
X = pca.transform(AE_data)

AE_data_test = np.zeros((8,696*3))
for i in range(8):
    idx = test_IDXS[i]
    for j in range(3):
        AE_data_test[i,j*696:(j+1)*696] = AE_all[idx*4+j+1]
x_test = pca.transform(AE_data_test)

X0MAX = X[:,0].max(); X0MIN = X[:,0].min()
X1MAX = X[:,1].max(); X1MIN = X[:,1].min()
x_rand = np.zeros((50,2))
for i in range(50):
    x0 = np.random.rand()*(X0MAX-X0MIN)+X0MIN
    x1 = np.random.rand()*(X1MAX-X1MIN)+X1MIN
    x_rand[i][0]=x0
    x_rand[i][1]=x1


plt.figure()
plt.scatter(X[:,0],X[:,1],c='b')
for i in range(20):
    idx = IDXS[i]
    az,ele = get_az_ele(idx)
    plt.text(X[i,0],X[i,1],'az'+str(az)+'_ele'+str(ele),size = 15)

plt.scatter(x_test[:,0],x_test[:,1],c='r')
for i in range(8):
    idx = test_IDXS[i]
    az,ele = get_az_ele(idx)
    plt.text(x_test[i,0],x_test[i,1],'az'+str(az)+'_ele'+str(ele),size = 15)

plt.scatter(x_rand[:,0],x_rand[:,1],c='g')
plt.xlabel('Score on PC 1 (82.43%)',fontsize=15,fontweight='bold')
plt.ylabel('Score on PC 2 (7.01%)',fontsize=15,fontweight='bold')
plt.legend(['Train angles','Test angles','Random Sampling'],fontsize=15)


#AEnew = pca.inverse_transform(X)
# [0.82431959 0.070132   0.04144262 0.02476158 0.00823945]
#print(pca.explained_variance_ratio_)
#np.random.choice(list(range(28)),20,replace = False)
#np.random.choice(list(range(28)),20)

"""
#find max AE range
for k in range(4):
    a = AE_all[[i*4+k for i in range(28)]]
    print(a.max(),a.min(),a.mean())
"""
#np.random.seed(372)
#AE_all = np.random.rand(112,696)
I_all = np.load('deal_with_data/I_all.npy')

# flux_all_all.shape: (28, 8, 696, 280)
# 无遮挡下，均取第四个对焦点的总能量，注意，尚未把直接照射到平板上而重复计算的695次减去。
flux_all_centeral = flux_all_all[:,4].sum(axis=2).sum(axis=1)
# 无遮挡下，均取第四个对焦点的总能量的最大值,共计28个数
flux_all_centeral_max = flux_all_all[:,4].sum(axis=1).max(axis=1)

flux_all_centeral_min = flux_all_all[:,4].sum(axis=1).min(axis=1)


ideal_flux_all = np.load('deal_with_data/ideal_flux_all.npy')
AE_all_aim = np.load('deal_with_data/AE_all_aim.npy')
#max_flux = {'':}
FLUX_MAX = 330000
for i in range(8):
    for j in range(112):
        pass
        #AE_all_aim[i,j] = max_norm(AE_all_aim[i,j].copy())
#idx:0-27
def get_idx(az,ele):
    if az==360:
        az = 0
    idx = az//30*4 + (ele-10)//20
    return idx

def get_ideal_flux_npa(az,ele):
    idx = get_idx(az,ele)
    return ideal_flux_all[idx].copy()

def get_flux_all_tmp(az,ele):
    idx = get_idx(az,ele)
    return flux_all_all[idx].copy()

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

def to_L_one(image_array):
    image_array*=255
    im = Image.fromarray(image_array)
    im = im.convert('L')
    return im

def show_one_pred(GA_flux, DL_flux,ref,solution,pred,i,more,save_dir = 'gray',add_name = ""):
    #im1 = to_L_one(ref)
    #im2 = to_L_one(solution)
    #a = im.size
    #im = im.resize((1*a[0], 1*a[1]), Image.ANTIALIAS)
    #im.show()
    if len(ref.shape)==2:
        d=1
        
    elif len(ref.shape)==3:
        #(2,70,17) (3,70,17)
        d=ref.shape[0]
    matplotlib.use('Agg')
    plt.figure(figsize=(20,10))
    plt.subplot(1,6,1)
    plt.title('input')
    #1d,realsolution
    if d==1:
        plt.imshow(ref,cmap='gray')
    #2d, bc
    elif d==2:
        plt.imshow(ref[1], cmap='gray')
    #3d,(3,70,17)-->(70,17,3)
    elif d==3:
        ref = ref.transpose(1,2,0)
        plt.imshow(ref/ref.max(), cmap = plt.cm.jet)
    #
    plt.subplot(1,6,2)
    plt.title('GA solution')
    #1d, output
    if len(solution.shape)==2:
        plt.imshow(solution, cmap='gray')
    else:
        solution = solution.transpose(1,2,0)
        plt.imshow(solution/solution.max(), cmap = plt.cm.jet)
        
    #3d, output
    ##
    plt.subplot(1,6,3)
    plt.title('DL solution (gen_image)')
    if len(pred.shape)==2:
        plt.imshow(pred, cmap='gray')
    else:
        pred = pred.transpose(1,2,0)
        plt.imshow(pred/pred.max(), cmap = plt.cm.jet)
    #plt.show()
    
    plt.subplot(1,6,4)
    plt.title('ref')
    #3d
    if d==3:
        #ref_true = to_2D(to_reflectivity(ref[:,:,0]))
        ref_true = to_2D(to_reflectivity(solution))
        plt.imshow(ref_true, cmap='gray')
    ###
    #2d, bc
    elif d==2:
        plt.imshow(ref[0], cmap='gray')
    #1d,realsolution
    elif d==1:
        plt.imshow(to_2D(to_reflectivity(ref)), cmap='gray')
    
    
    plt.subplot(1,6,5)
    plt.title('GA flux')
    plt.imshow(GA_flux, cmap=plt.cm.jet)
    
    plt.subplot(1,6,6)
    plt.title('DL flux')
    plt.imshow(DL_flux, cmap=plt.cm.jet)
    
    
    if add_name == "":
        figname = str(i+1)+'_more'+str(more)+'.png'
    else:
        figname = add_name+'_more'+str(more)+'.png'
    savepath = os.path.join(save_dir,figname)
    plt.savefig(savepath)


def show_one_pred_new(DL_flux,ref,pred,i,save_dir = 'gray',add_name = ""):
    #im1 = to_L_one(ref)
    #im2 = to_L_one(solution)
    #a = im.size
    #im = im.resize((1*a[0], 1*a[1]), Image.ANTIALIAS)
    #im.show()
    if len(ref.shape)==2:
        d=1
        
    elif len(ref.shape)==3:
        #(2,70,17) (3,70,17)
        d=ref.shape[0]
    matplotlib.use('Agg')
    plt.figure(figsize=(20,10))
    plt.subplot(1,3,1)
    plt.title('input')
    #1d,realsolution
    if d==1:
        plt.imshow(ref,cmap='gray')
    #2d, bc
    elif d==2:
        plt.imshow(ref[1], cmap='gray')
    #3d,(3,70,17)-->(70,17,3)
    elif d==3:
        ref = ref.transpose(1,2,0)
        plt.imshow(ref/ref.max(), cmap = plt.cm.jet)
    #

    #3d, output
    ##
    plt.subplot(1,3,2)
    plt.title('DL solution (gen_image)')
    if len(pred.shape)==2:
        plt.imshow(pred, cmap='gray')
    else:
        pred = pred.transpose(1,2,0)
        plt.imshow(pred/pred.max(), cmap = plt.cm.jet)
    #plt.show()

    
    plt.subplot(1,3,3)
    plt.title('DL flux')
    plt.imshow(DL_flux, cmap=plt.cm.jet)
    
    
    if add_name == "":
        figname = str(i+1)+'.png'
    else:
        figname = add_name+'.png'
    savepath = os.path.join(save_dir,figname)
    print(savepath)
    plt.savefig(savepath)




def to_reflectivity(im_1d):
    v0 = to_vector(im_1d)
    reflectivity = np.empty([696],dtype=np.float32)
    for i in range(696):
        #惨痛教训，要记得查看：
        #pix2pix,regression
        if abs(v0[i])>10e-5:
        #if abs(v0[i])>0.011:
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


#v3 is added
#def calculate_one(v3, az,ele,im0,im1,im2,i,pr=True,pandas=False):
def calculate_one(central_power, az,ele,im0,im1,im2,i,pr=True,pandas=False):
    #f = open("dis1.txt","a")
    #f.write('test'+str(i+1)+'\n')
    if az==360:
        az=0
    if pr==True:
        print('test'+str(i+1)+'\n')
    v1 = to_vector(im1*20+170)
    v2 = to_vector(im2*20+170)
    v0 = to_reflectivity(im1)
    bestscore,bestpower,bestuniformity,GA_flux = get_one_fitness(az,ele,v1,v0)

    v2 = standard(v2.copy())
    score,power,uniformity,DL_flux = get_one_fitness(az,ele,v2,v0)
    
    power_more = 1 - power/bestpower
    uniformity_more = 1 - bestuniformity/uniformity
    more = round(((power_more*2+uniformity_more)/3)*100,2)#
    more2 = (bestscore-score)/1e10
    
    if pr==True:
        print("GA\t"+"%e\t"%bestscore+"%e\t"%bestpower+str(round(bestuniformity,6)))
        print('intercept(截断率)：',bestpower/central_power,'\n')
        print("pix2pix\t"+"%e\t"%score+"%e\t"%power+str(round(uniformity,6)))
        print('intercept(截断率)：',power/central_power,'\n')
        print(str(more)+" percent less")
        print(str(more2)+" percent more",'\n')
    
    if pandas==True:
        return v0,v1,v2,bestscore,bestpower,bestuniformity,score,power,uniformity,flux_together,more
        #return aa,bb,cc,bestscore,bestpower,bestuniformity,score,power,uniformity,flux_together,more
    else:
        return more,GA_flux,DL_flux


def standard(v2):
    for i in range(len(v2)):
        label = round((v2[i]-171.25)/2.5)
        if label>=8:
            label=7
        elif label<0:
            label=0
        v2[i] = 171.25 + label*2.5
    return v2





#展现所有GA图像
def to_L(filename,scale):
    ref,solution = get_dataset_2D(scale,filename)
    for i in range(scale):
        show_one_index(ref[i],solution[i],i)



def get_str(ts):
    npa = ts.detach().numpy()
    npa = npa*20+170
    return ','.join([str(i) for i in npa])


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
'''
def to_vector(two_dimension):
    return two_dimension.copy().reshape(696)
'''
def to_vector(two_dimension):
    #h_num_npa = get_h_num_in_one_r()
    used = 0
    width = np.max(h_num_npa)
    #vector = np.empty([696],dtype=np.float32)
    vector = np.array([],dtype=np.float32)
    for i in range(len(h_num_npa)):
        a = h_num_npa[i]
        tmp = two_dimension[i]
        core = tmp
        pad = int((width-a)/2)
        vector = np.append(vector,core[pad:width-pad])
    return vector


#1*696===》17*70

def to_2D(one_dimension_npa):#(1,696)
    #h_num_npa = get_h_num_in_one_r()
    used = 0
    width = np.max(h_num_npa)
    #two_dimension = np.empty([0,width*2], dtype=np.float32)
    two_dimension = np.empty([0,width], dtype=np.float32)
    for i in range(len(h_num_npa)):
        a = h_num_npa[i]
        tmp_npa = np.zeros(width)
        pad = int((width-a)/2)
        tmp_npa[pad:width-pad]= one_dimension_npa[used:used+a]
        two_dimension = np.append(two_dimension,[tmp_npa], axis=0)
        used+=a
    return two_dimension.astype(dtype=np.float32)#(140，34)


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
        

def get_dataset(filename):
    scale = get_scale(filename)
    txtname_all = []
    az_all = []
    ele_all = []
    ref = np.empty([0,696],dtype=np.float32)
    solution = np.empty([0,696],dtype=np.float32)
    #result_evaluate = []
    with open(filename,"r") as f:
        for i in range(scale):
            #line1
            a = f.readline()
            az = int(a[a.find('az')+2 : a.find('_ele')])
            ele = int(a[a.find('ele')+3 : a.find('.txt')])
            az_all.append(az)
            ele_all.append(ele)
            #txtname = f.readline()
            txtname_all.append(a)
            #line2
            ref_tmp_str = f.readline().split(',')
            ref_tmp = np.array([float(_) for _ in ref_tmp_str],dtype=np.float32)
            if len(ref_tmp) != 696:
                print("REF ERROR",i)
                return
            ref = np.append(ref,[ref_tmp],axis=0)
            #line3
            gen_num = f.readline()
            #line4
            result_evaluate_tmp = f.readline()
            #result_evaluate.append(result_evaluate_tmp)
            #line5
            solution_tmp_str = f.readline().split(',')
            solution_tmp = np.array([float(_) for _ in solution_tmp_str],dtype=np.float32)
            
            ######需要特别注意，与pix2pix中的deal_with相比，这一句被删除了
            #solution_tmp = (solution_tmp-170)/20
            ##########
            #solution_tmp = (solution_tmp+17.5)/35
            
            if len(solution_tmp) != 696:
                print("SOLUTION ERROR",i)
                return
            solution = np.append(solution,[solution_tmp],axis=0)
            #line6
            f.readline()
    return ref,solution,az_all,ele_all,txtname_all#result_evaluate


resize2 = transforms.Resize([70,17])
resize22 = transforms.Resize([320,56])
resize26 = transforms.Resize([40,7])
resize23 = transforms.Resize([160,28])

class MyDataset(Dataset):
    def __init__(self, filename,inputdim=1,outputdim=1,type11='b', maxnum=200000, need_flux=False):
        # 这个list存放所有图像的地址
        dsname = filename[:filename.find('.txt')]
        rootpath = os.path.join('dataset_np', dsname)
        if not os.path.exists(rootpath):
            os.mkdir(rootpath)
            result = get_dataset(filename)
            #self.ref = np.array(zip(result[0],raw[0]),dtype=object)
            self.ref = result[0]
            #self.ref = list(zip(result[0],raw[0]))
            self.solution = result[1]
            self.az_all = result[2]
            self.ele_all = result[3]
            self.txtname_all = result[4]
            
            #self.scale = get_scale(filename)
            np.save(os.path.join('dataset_np',dsname,'ref.npy'), self.ref)
            np.save(os.path.join('dataset_np',dsname,'solution.npy'), self.solution)
            np.save(os.path.join('dataset_np',dsname,'az_all.npy'), self.az_all)
            np.save(os.path.join('dataset_np',dsname,'ele_all.npy'), self.ele_all)
            f = open(os.path.join('dataset_np',dsname,'txtname_all.txt'),'w')
            f.writelines(self.txtname_all)
            f.close()
        else:
            self.ref = np.load(os.path.join('dataset_np',dsname,'ref.npy'))
            self.solution = np.load(os.path.join('dataset_np',dsname,'solution.npy'))
            self.az_all = np.load(os.path.join('dataset_np',dsname,'az_all.npy'))
            self.ele_all = np.load(os.path.join('dataset_np',dsname,'ele_all.npy'))
            #详细计算过程参看 calculate_flux_dataset.py
            
            if need_flux == True:
                #遗传算法最优解对应光斑
                self.flux_all = np.load(os.path.join('dataset_np',dsname,'flux_all.npy'))
                self.flux_all /= FLUX_MAX
                #按照无遮挡情形构造光斑
                noshadow_flux = np.zeros(self.flux_all.shape)
                for az in range(0,181,30):
                    for ele in range(10,71,20):
                        #np.tile复制
                        #tmp_copy = np.tile((get_ideal_flux_npa(az,ele)).astype(dtype=np.float32), (np.sum((self.az_all==az) & (self.ele_all==ele)),1))
                        #noshadow_flux[(self.az_all==az) & (self.ele_all==ele),:] = tmp_copy
                        #传播：broadcast
                        noshadow_flux[(self.az_all==az) & (self.ele_all==ele),:] = (get_ideal_flux_npa(az,ele))
                        
                self.noshadow_flux = noshadow_flux/FLUX_MAX
                #按照和无遮挡情形成比例构造光斑
                #lambd表征该种情形的阴影所引起的总能量减少，进而设定的AFD也应该减少
                #self.lambds = np.ones(len(self.az_all))
                
                #real_lambds扣除了（90,180]的直射到吸热器的增益。
                self.real_lambds = np.ones(len(self.az_all))
                
                lambda_flux = np.zeros(self.flux_all.shape)
                for i in range(lambda_flux.shape[0]):
                    az,ele = self.az_all[i], self.ele_all[i]
                    idx = get_idx(az,ele)
                    
                    #lambd = np.sum(I_all[idx]*self.ref[i]) / np.sum(I_all[idx])
                    #self.lambds[i] = lambd
                    
                    ##############################
                    #计算真实的由遮挡引起的lambds系数#
                    ##############################
                    #az = 120,150,180时需要扣除
                    if abs(az-180) <= 90 and ele < 90:
                        more = 1000* math.cos((az-180)/180*math.pi) * math.cos(ele/180*math.pi) *280*( np.sum(self.ref[i])-1 )
                    else:
                        more = 0
                    real_lambd = (np.sum(I_all[idx]*self.ref[i]) - more) / (np.sum(I_all[idx]) - more)
                    self.real_lambds[i] = real_lambd
                    #############################
                    
                    #lambd = (1+lambd)/2
                    #lambd = 1
                    #lambda_flux[i] = (lambd * get_ideal_flux_npa(az,ele)).astype(dtype=np.float32)
                    lambda_flux[i] = (real_lambd * get_ideal_flux_npa(az,ele)).astype(dtype=np.float32)
                self.lambda_flux = lambda_flux/FLUX_MAX
                
                
                
            ######
            f = open(os.path.join('dataset_np',dsname,'txtname_all.txt'),'r')
            self.txtname_all = f.readlines()
            f.close()        
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
        #index = 1000
        #缩小数据量的简便办法
        #index = 10000+index%100
        #index = 10000
        ref,solution,az,ele,txtname = self.ref[index],self.solution[index],self.az_all[index],self.ele_all[index],self.txtname_all[index]
        #(1,1).aa: 
        #input:raw_solution under no_shadow condition
        #(1,1).b:
        #未能区分不同的太阳角度，舍去
        #input.bb: ref, output:solution
        #(2,1).:
        #input: ref, (az,ele cross), output:solution
        
        # use all data
        idx = get_idx(az,ele)
        #sample 20
        #idx = np.random.choice(IDXS)
        #az,ele = get_az_ele(idx)
        
        #
        
        #基准AFD的计算方案一:(均值方案)
        central_flux = flux_all_centeral[idx]# * self.lambds[index]
        if abs(az-180) <= 90 and ele < 90:
            central_flux = central_flux- 1000* math.cos((az-180)/180*math.pi) * math.cos(ele/180*math.pi) \
                *280*( np.sum(ref)-1 )
        #最终所得central_flux表示 按照中心聚焦策略所得光斑 的 均值 计算出的基准AFD，
        #lambds考虑了不同阴影大小的影响。
        #其乘上1.3左右即为一个具有约束的AFD设定值，系数越大，AFD约束越小;
        central_flux = central_flux*self.real_lambds[index]
        
        new_ref,new_solution,new_intensity = MyTransform(index, ref.copy(),solution.copy(),az,ele,
                                                         self.inputdim,self.outputdim,
                                                         needweight=True,
                                                         type11=self.type11,
                                                         need_flux=self.need_flux)
        ref_2D = np.empty([1,70,17],dtype=np.float32)
        ref_2D[0] = to_2D(ref)
        if self.need_flux==True:
            new_flux = self.flux_all[index].astype(dtype=np.float32)
            new_flux1 = torch.tensor(new_flux).reshape(40,7).unsqueeze(0)
            #print(new_flux1.shape)
            #40,7==>70,17, not necessary
            #new_flux = resize2(new_flux).squeeze()#*1.05#1.1
            new_flux = resize26(new_flux1).squeeze()
            #new_ref_rand = np.zeros([4,70,17],dtype=np.float32)
            #方案一：按行随机排列
            #for i in range(70):
            #    raw_idx = np.where(new_ref[1,i]>0)[0]
            #    rand_idx = np.random.permutation(raw_idx)
            #    new_ref_rand[:,i,raw_idx] = new_ref[:,i,rand_idx]
            
            #方案二：所有非零元素随机排列
            '''
            raw_idx = np.where(new_ref[1]>0)
            raw_idx_x = raw_idx[0]
            raw_idx_y = raw_idx[1]
            new_index = np.random.permutation(len(raw_idx_x))
            rand_idx_x = raw_idx_x[new_index]
            rand_idx_y = raw_idx_y[new_index]
            new_ref_rand[:,raw_idx_x,raw_idx_y] = new_ref[:,rand_idx_x,rand_idx_y]
            '''    
            #方案三：所有非零元素随机排列
            #在def to_2D 中，初值为np.zeros()-0.01
            '''
            raw_idx = np.where(new_ref[1]>0.001)
            raw_idx_x = raw_idx[0]
            raw_idx_y = raw_idx[1]
            new_index = np.random.permutation(len(raw_idx_x))
            rand_idx_x = raw_idx_x[new_index]
            rand_idx_y = raw_idx_y[new_index]
            new_ref_rand[:,raw_idx_x,raw_idx_y] = new_ref[:,rand_idx_x,rand_idx_y]
            '''
            
            ######
            
            return new_ref, new_solution, new_flux, new_intensity, ref_2D, txtname,az,ele,central_flux
            #return new_ref_rand, new_solution, new_flux, new_intensity, ref_2D, txtname,az,ele,central_flux
            #return raw_idx_x,raw_idx_y,rand_idx_x,rand_idx_y, new_ref_rand, new_solution, new_flux, new_intensity, ref_2D, txtname,az,ele,central_flux
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

#index 仅仅用于计算随机数种子
def MyTransform(index, ref,solution,az,ele,inputdim=4,outputdim=1,needweight=False,type11 = 'a', need_flux=False):
    if type11=='raw':
        return ref,solution,[]
    #if need_flux==True:
    #    outputdim = 1
    #    inputdim = 4
        
    if az==360:
        az=0 
    #0-13
    idx = get_idx(az,ele)%14
    zero_idx = np.where(ref==0)
    if inputdim==4 and type11=='AE':
        d = 4 
        #part1.transform ref, 
        #one possible ref, by autoencoder
        AE_1 = np.empty([d,696],dtype=np.float32)
        AE_1_tmp = AE_all[idx*(d)+1 : (idx+1)*(d)].copy()
        AE_1[0:d-1,:] = AE_1_tmp
        #assign2
        AE_1[d-1,:] = I_all[idx].copy()
        #AE_1[d-1,zero_idx] = 0
        for j in range(d):
            #AE_1[j,0:696] = norm(AE_1[j,0:696])
            AE_1[j,0:696] = max_norm(AE_1[j,0:696])
        AE_1[:,zero_idx] = 0
        new_ref = np.empty([d,70,17],dtype=np.float32)
        for k in range(d):
            new_ref[k] = to_2D(AE_1[k])
    elif inputdim==3 and type11=='AE':
        d = 3
        #part1.transform ref, 
        #one possible ref, by autoencoder
        AE_1 = np.empty([d,696],dtype=np.float32)
        
        #1.raw AE code
        #AE_1_tmp = AE_all[idx*(4)+1 : (idx+1)*(4)].copy()
        #AE_1[0:d,:] = AE_1_tmp
        #######
        #2.pure random generate,43,22,36
        #log1020exp, 70,40,70
        #np.random.seed(index%10)
        
        #AE_1[0] = np.random.rand(696)*70
        #AE_1[1] = np.random.rand(696)*35
        #AE_1[2] = np.random.rand(696)*70
        
        
        ######
        x0 = np.random.rand()*(X0MAX-X0MIN)+X0MIN
        x1 = np.random.rand()*(X1MAX-X1MIN)+X1MIN
        x = [x0,x1]
        AEnew = pca.inverse_transform(x)
        for j in range(3):
            AE_1[j] = AEnew[j*696:(j+1)*696]
        ######
        #3.random in the basis of 28 real solar angle, 10%
        '''
        AE_1_tmp = AE_all[idx*(4)+1 : (idx+1)*(4)].copy()
        AE_1[0:d,:] = AE_1_tmp
        AE_1[0] += (np.random.rand(696)-0.5)*4.3*2
        AE_1[1] += (np.random.rand(696)-0.5)*2.2*2
        AE_1[2] += (np.random.rand(696)-0.5)*3.6*2
        '''
        for j in range(d):
            #AE_1[j,0:696] = norm(AE_1[j,0:696])
            #AE_1[j,0:696] = max_norm(AE_1[j,0:696])
            pass
        AE_1[:,zero_idx] = 0
        new_ref = np.empty([d,70,17],dtype=np.float32)
        for k in range(d):
            new_ref[k] = to_2D(AE_1[k])
    elif inputdim==280 and type11=='AE':
        d = 280
        #part1.transform ref, 
        #one possible ref, by autoencoder
        AE_1 = np.empty([d,696],dtype=np.float32)
        rand_list = np.random.randint(0,8,696)
        
        AE_1_tmp = np.transpose(flux_all_all[idx,rand_list,np.arange(696),:].copy())
        AE_1[0:d,:] = AE_1_tmp
        for j in range(d):
            #AE_1[j,0:696] = norm(AE_1[j,0:696])
            AE_1[j,0:696] = max_norm(AE_1[j,0:696])
        AE_1[:,zero_idx] = 0
        new_ref = np.empty([d,70,17],dtype=np.float32)
        for k in range(d):
            new_ref[k] = to_2D(AE_1[k])
    ################################################    
    #part2.transform solution
    
    #new_solution = np.empty([696],dtype=np.float32)
    #1d
    if outputdim==1:
        new_solution = (solution.copy()-170)/20
        new_solution[zero_idx] = 0
        #raw
        new_solution = to_2D(new_solution)

    if needweight==True:
        #intensity_1 = np.empty([696],dtype=np.float32)
        intensity_1 = I_all[idx].copy()
        intensity_1[zero_idx] = 0
        intensity_1 = norm(intensity_1)
        #new_intensity = np.empty([1,70,17],dtype=np.float32)
        new_intensity = to_2D(intensity_1)
        return new_ref, new_solution, new_intensity
    else:
        return new_ref, new_solution

