valid:
1.files:
test_G_1.py: test GA, Two-Stage model, and E2E model on given 28 solar angles.
test_G_2.py: test GA, E2E model, and Robust E2E model trained on 20 solar angles, and tested on 8 unseen solar angles.
model.py: pix2pix model (including 'Generator' and 'Discriminator') and convolutional autoencoder ('AE_conv')
deal_with.py: many useful functions.
----------------------------------------------------------------------------------------------
2.folders:
2a. data
deal_with_data: the simulation data used by 'deal_with.py'. 
       'flux_all_all.npy' is the raw simulation data, its size is (28,8,696,280), 
       which means 28 solar angles, 8 aiming points, 696 heliostats, and 280 is the cell number of the receiver.
dataset_np: GA benchmark dataset, including training and testing datasets.
2b. models and results 
test_results: 2 tables of 'test_G_1.py' and 'test_G_2.py'.
'G_test_***': Some comparison charts.
'Two_Stage_model_on_given_flux_maps', 'E2E_model_on_given_flux_maps', \
'E2E_model_on_unseen_flux_maps','Robust_E2E_model_on_unseen_flux_maps': several pre-trained models.
