train:
1.files:
Robust_E2E_train.py: code for traing robust end-to-end model.
model.py: pix2pix model (including 'Generator' and 'Discriminator') and convolutional autoencoder ('AE_conv')
deal_with.py: many useful functions.
----------------------------------------------------------------------------------------------
2.folders:
2a. data
deal_with_data: the simulation data used by 'deal_with.py'. 
       'flux_all_all.npy' is the raw simulation data, its size is (28,8,696,280), 
       which means 28 solar angles, 8 aiming points, 696 heliostats, and 280 is the cell number of the receiver.
dataset_np: GA benchmark dataset, including training and testing datasets.
2b. simple models 
AEmodels: pretrained convolutional autoencoder model.
onehot_sigmoid.pt: pretrained transform network (a simple ANN model with sigmoid function) as mentioned in the article. 
2c. training process
'gray...': plot the solution of DL each 10 steps, which shows how the predictive aiming strategy can be optimized step by step.
'check_points...': save DL models.
