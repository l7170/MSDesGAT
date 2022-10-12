# -*- coding: utf-8 -*-
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import HSI_SELF
import random
from sklearn import preprocessing, metrics, svm, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import comparsion_nn
from torch.utils.data import DataLoader
from PIL import Image
import math
import pandas as pd

dataset_name = 'Indian_pines'
learning_rate = 5e-4
train_ratio = 0.01
val_ratio = 0.01
if dataset_name == 'Pavia_University':
    groundtruth_filename = './/datasets//PaviaU_gt.mat'
    groundtruth_mat = 'paviaU_gt'
    data_filename = './/datasets//PaviaU.mat'
    HSI_mat = 'paviaU'
    SSConv_kernel = 3
    SSConv_num_depth_conv_layer = 2
    RGB_bands = np.array([55, 41, 12])
    segs_list = [7000, 6000, 5000, 4000, 3000] #0.95446044[7000, 6000, 5000, 4000, 3000]
    self_Thres = 0.995
    max_epoch = 600
    learning_rate = 1e-3
elif dataset_name == 'Indian_pines':
    groundtruth_filename = './/datasets//Indian_pines_gt.mat'
    groundtruth_mat = 'indian_pines_gt'
    data_filename = './/datasets//Indian_pines_corrected.mat'
    HSI_mat = 'indian_pines_corrected'
    SSConv_kernel = 5
    SSConv_num_depth_conv_layer = 1
    RGB_bands = np.array([43, 21, 11])
    segs_list =[1600, 800, 400, 200, 100] #[2600, 2300, 2000]#[50,250,450,650,850,1050,1250,1450,1650,1850]
    max_epoch = 600

    self_Thres = 0.95
    train_ratio = 0.05
    val_ratio = 0.05
elif dataset_name == 'WHU_HongHu':
    groundtruth_filename = './/datasets//WHU_Hi_HongHu_gt.mat'
    groundtruth_mat = 'WHU_Hi_HongHu_gt'
    data_filename = './/datasets//WHU_Hi_HongHu.mat'
    HSI_mat = 'WHU_Hi_HongHu'
    SSConv_kernel = 3
    SSConv_num_depth_conv_layer = 3
    RGB_bands = np.array([55, 41, 12])
    segs_list = [2600, 2500, 2400]
    self_Thres = 0.995
    max_epoch = 600
elif dataset_name == 'Salinas':
    groundtruth_filename = './/datasets//Salinas_gt.mat'
    groundtruth_mat = 'salinas_gt'
    data_filename = './/datasets//Salinas_corrected.mat'
    HSI_mat = 'salinas_corrected'
    SSConv_kernel = 5
    SSConv_num_depth_conv_layer = 1
    RGB_bands = np.array([55, 41, 12])
    segs_list = [2400, 2300, 2200, 2100, 2000]
    self_Thres = 0.995
    max_epoch = 600

labelfile = scio.loadmat(groundtruth_filename)  # 'PaviaU_gt.mat'Indian_pines_gt
Groundtruth = labelfile.get(groundtruth_mat)  # paviaU_gt indian_pines_gt
Height, Width = Groundtruth.shape
HSIfile = scio.loadmat(data_filename)  # PaviaU.mat Indian_pines_corrected
HSI = HSIfile.get(HSI_mat)  # paviaU indian_pines_corrected
Bands = HSI.shape[2]

data = np.reshape(HSI, [Height * Width, Bands])
minMax = preprocessing.StandardScaler()
data = minMax.fit_transform(data)
samples_type = 'ratio'

class_count = int(np.amax(Groundtruth))
train_rand_idx = np.empty((0, 2), dtype='int')
val_rand_idx = np.empty((0, 2), dtype='int')
test_idx = np.empty((0, 2), dtype='int')
train_label_weight = np.zeros(class_count)
val_label_weight = np.zeros(class_count)
test_label_weight = np.zeros(class_count)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

Groundtruth = np.reshape(Groundtruth, [Height * Width])
random.seed(200)
np.random.seed(200)
torch.cuda.manual_seed(200)

if samples_type == 'ratio':
    for i in range(class_count):
        idx = np.where(Groundtruth == i + 1)[-1]
        label_tmp = np.ones(len(idx), dtype='int') * (i + 1)
        tmp = np.concatenate((idx[:][np.newaxis].T, label_tmp[np.newaxis].T), axis=1)
        test_idx = np.concatenate((test_idx, tmp), axis=0)
        test_label_weight[i] = 1. / (len(idx) + 1e-4)

        samplesCount = len(idx)
        rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
        rand_idx = random.sample(rand_list,
                                 (np.ceil(samplesCount * train_ratio) + np.ceil(samplesCount * val_ratio)).astype(
                                     'int32'))  # 随机数数量 四舍五入(改为上取整)
        rand_real_idx_per_class = idx[rand_idx]
        label_tmp = np.ones(len(idx[rand_idx]), dtype='int') * (i + 1)
        tmp = np.concatenate((idx[rand_idx][np.newaxis].T, label_tmp[np.newaxis].T), axis=1)
        train_rand_idx = np.concatenate((train_rand_idx, tmp[:int(np.ceil(samplesCount * train_ratio)), :]), axis=0)
        val_rand_idx = np.concatenate((val_rand_idx, tmp[int(np.ceil(samplesCount * train_ratio)):, :]), axis=0)
        train_label_weight[i] = 1. / (np.ceil(samplesCount * train_ratio) + 1.)
        val_label_weight[i] = 1. / (np.ceil(samplesCount * val_ratio) + 1.)

train_label_mask = np.zeros(Height * Width, dtype='int')
train_label_mask[train_rand_idx[:, 0]] = np.ones(train_rand_idx.shape[0])
train_label_mat = np.zeros(Height * Width, dtype='int')
train_label_mat[train_rand_idx[:, 0]] = train_rand_idx[:, 1]
train_label_mat = np.reshape(train_label_mat, [Height, Width])
scio.savemat('Train_gt.mat', {'TRAIN_GT': train_label_mat})

val_label_mask = np.zeros(Height * Width, dtype='int')
val_label_mask[val_rand_idx[:, 0]] = np.ones(val_rand_idx.shape[0])

test_label_mask = np.zeros(Height * Width, dtype='int')
test_label_mask[test_idx[:, 0]] = np.ones(test_idx.shape[0])

Y = (Groundtruth - 1).astype(np.int32)

##SVM
# clf = svm.SVC(class_weight='balanced')
# SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
#                                        'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
#                    {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]
# X=data[train_label_mask.astype(bool),:]
# label=Y[train_label_mask.astype(bool)]
# clf = model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, verbose=5, n_jobs=1)
# clf.fit(X, label)
# print("SVM best parameters : {}".format(clf.best_params_))
# prediction = clf.predict(data[test_label_mask.astype(bool),:])
# OA = metrics.accuracy_score( Y[test_label_mask.astype(bool)], prediction)
# kappa = metrics.cohen_kappa_score(Y[test_label_mask.astype(bool)], prediction)
# confusion_matrix = metrics.confusion_matrix(Y[test_label_mask.astype(bool)], prediction)
# producer_acc = np.diag(confusion_matrix)/np.sum(confusion_matrix,axis=1)
# AA = np.average(producer_acc)
# print("SVM: \n", "test OA=", OA, ' test AA=', AA,' kpp=', kappa,
#       '\nconfusion matrix=', confusion_matrix, "\nproducer_acc", producer_acc)
# HSI_SELF.write_acc(OA, AA, producer_acc, kappa, confusion_matrix,
#                    'results//' + dataset_name+ '_results.txt', 'SVM')
# classification_map = clf.predict(data)
# classification_map = classification_map.reshape([Height, Width])
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//SVM-" + dataset_name + "-" + str(OA))
#
# ###Random Forest
# clf = RandomForestClassifier(class_weight='balanced')
# X=data[train_label_mask.astype(bool),:]
# label=Y[train_label_mask.astype(bool)]
# clf.fit(X, label)
# prediction = clf.predict(data[test_label_mask.astype(bool),:])
# OA = metrics.accuracy_score( Y[test_label_mask.astype(bool)], prediction)
# kappa = metrics.cohen_kappa_score(Y[test_label_mask.astype(bool)], prediction)
# confusion_matrix = metrics.confusion_matrix(Y[test_label_mask.astype(bool)], prediction)
# producer_acc = np.diag(confusion_matrix)/np.sum(confusion_matrix, axis=1)
# AA = np.average(producer_acc)
# print("Random Forest: \n", "test OA=", OA, 'test AA=', AA,'kpp=', kappa, '\nconfusion matrix=',
#       confusion_matrix, "\nproducer_acc", producer_acc)
# HSI_SELF.write_acc(OA, AA, producer_acc, kappa, confusion_matrix,
#                    'results//' + dataset_name+ '_results.txt', 'Random_Forest')
# classification_map = clf.predict(data)
# classification_map = classification_map.reshape([Height, Width])
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//Random_Forest-" + dataset_name + "-" + str(OA))
#
# ################################# 1D CNN starting
#
# train_dataset = comparsion_nn.HSI1D_dataset(data[train_label_mask.astype(bool),:],
#                                             Y[train_label_mask.astype(bool)])
#
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#
# val_dataset = comparsion_nn.HSI1D_dataset(data[val_label_mask.astype(bool),:],
#                                             Y[val_label_mask.astype(bool)])
# val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
#
# test_dataset = comparsion_nn.HSI1D_dataset(data, Y)
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
# #### baseline
# method = 'Baseline'
# net = comparsion_nn.Baseline(Bands, class_count, dropout=False)
# print(net)
# net.to(device)
# torch.cuda.empty_cache()
# net = comparsion_nn.train_model(net, train_loader, val_loader, 1e-3, train_label_weight.astype('float32'), device, num_epochs=100)
# OA, classification_map = comparsion_nn.test_model(net, test_loader, test_label_mask.astype('bool'),
#                                                   Height, Width, device,dataset_name, method)
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//Baseline-" + dataset_name + "-" + str(OA))
#
# ###MouEtAl
# method = 'MouEtAl'
# net = comparsion_nn.MouEtAl(Bands, class_count)
# print(net)
# net.to(device)
# torch.cuda.empty_cache()
# net = comparsion_nn.train_model(net, train_loader, val_loader, 0.01, train_label_weight.astype('float32'), device, num_epochs=100)
# OA, classification_map = comparsion_nn.test_model(net, test_loader, test_label_mask.astype('bool'),
#                                                   Height, Width, device,dataset_name, method)
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//MouEtAl-" + dataset_name + "-" + str(OA))
#
# ###HuEtAl
# method= 'HuEtAl'
# net = comparsion_nn.HuEtAl(Bands, class_count)
# print(net)
# net.to(device)
# torch.cuda.empty_cache()
# net = comparsion_nn.train_model(net, train_loader, val_loader, 1e-3, train_label_weight.astype('float32'), device, num_epochs=100)
# OA, classification_map = comparsion_nn.test_model(net, test_loader, test_label_mask.astype('bool'),
#                                                   Height, Width, device,dataset_name, method)
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//HuEtAl-" + dataset_name + "-" + str(OA))
#
# ######################### 2D,3D CNN starting
# ### He_etal
# method = 'HeEtAl'
# HSI_n = np.reshape(data, [Height, Width, -1])
# patch_size = 7
# train_dataset = comparsion_nn.HSI2D_dataset(HSI_n, train_rand_idx, patch_size=patch_size,
#                                             flip_augmentation=True, radiation_augmentation=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataset = comparsion_nn.HSI2D_dataset(HSI_n, val_rand_idx, patch_size=patch_size)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
# test_dataset = comparsion_nn.HSI2D_dataset(HSI_n, np.concatenate((np.arange(Height*Width)[None],Groundtruth[None]),axis =0).T, patch_size=patch_size)
# test_loader = DataLoader(test_dataset, batch_size=64)
# net = comparsion_nn.HeEtAl(Bands, class_count, patch_size=patch_size)
# print(net)
# net.to(device)
# torch.cuda.empty_cache()
# net = comparsion_nn.train_model(net, train_loader, val_loader, 5e-4, train_label_weight.astype('float32'), device, num_epochs=100)
# OA, classification_map = comparsion_nn.test_model(net, test_loader, test_label_mask.astype('bool'),
#                                                   Height, Width, device,dataset_name, method)
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//HeEtAl-" + dataset_name + "-" + str(OA))
# ###Li_etal
# method = 'LiEtAl'
# HSI_n = np.reshape(data, [Height, Width, -1])
# patch_size = 5
# train_dataset = comparsion_nn.HSI2D_dataset(HSI_n, train_rand_idx, patch_size=patch_size,
#                                             flip_augmentation=True, radiation_augmentation=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataset = comparsion_nn.HSI2D_dataset(HSI_n, val_rand_idx, patch_size=patch_size)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
# test_dataset = comparsion_nn.HSI2D_dataset(HSI_n, np.concatenate((np.arange(Height*Width)[None],Groundtruth[None]),axis =0).T, patch_size=patch_size)
# test_loader = DataLoader(test_dataset, batch_size=64)
# net = comparsion_nn.LiEtAl(Bands, class_count, patch_size=patch_size)
# print(net)
# net.to(device)
# torch.cuda.empty_cache()
# net = comparsion_nn.train_model(net, train_loader, val_loader, 1e-3, train_label_weight.astype('float32'), device, num_epochs=100)
# OA, classification_map = comparsion_nn.test_model(net, test_loader, test_label_mask.astype('bool'),
#                                                   Height, Width, device,dataset_name, method)
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//LiEtAl-" + dataset_name + "-" + str(OA))
# # DCNN 2DCNN
# method = 'DCNN'
# pca = PCA(svd_solver='randomized')
# pca.fit(data)
# rPCA_Bands = np.where(np.cumsum(pca.explained_variance_ratio_)>0.99)[0][0]
# pca = PCA(n_components=rPCA_Bands, svd_solver='randomized')
# data_pca = pca.fit_transform(data)
# HSI_n = np.reshape(data_pca, [Height, Width, -1])
# patch_size = 5
# train_dataset = comparsion_nn.HSI2D_dataset(HSI_n, train_rand_idx, patch_size=patch_size,
#                                             flip_augmentation=True, radiation_augmentation=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataset = comparsion_nn.HSI2D_dataset(HSI_n, val_rand_idx, patch_size=patch_size)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
# test_dataset = comparsion_nn.HSI2D_dataset(HSI_n, np.concatenate((np.arange(Height*Width)[None],Groundtruth[None]),axis =0).T, patch_size=patch_size)
# test_loader = DataLoader(test_dataset, batch_size=64)
# net = comparsion_nn.DCNN(rPCA_Bands, class_count, patch_size=patch_size)
# print(net)
# net.to(device)
# torch.cuda.empty_cache()
# net = comparsion_nn.train_model(net, train_loader, val_loader, 5e-4, train_label_weight.astype('float32'), device, num_epochs=100)
# OA, classification_map = comparsion_nn.test_model(net, test_loader, test_label_mask.astype('bool'),
#                                                   Height, Width, device,dataset_name, method)
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//DCNN-" + dataset_name + "-" + str(OA))
# #### HybirdCNN
# method = 'HybirdCNN'
#
# HSI_n = np.reshape(data, [Height, Width, -1])
# patch_size = 11
# train_dataset = comparsion_nn.HSI2D_dataset(HSI_n, train_rand_idx, patch_size=patch_size,
#                                             flip_augmentation=True, radiation_augmentation=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataset = comparsion_nn.HSI2D_dataset(HSI_n, val_rand_idx, patch_size=patch_size)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
# test_dataset = comparsion_nn.HSI2D_dataset(HSI_n, np.concatenate((np.arange(Height*Width)[None],Groundtruth[None]), axis=0).T, patch_size=patch_size)
# test_loader = DataLoader(test_dataset, batch_size=64)
# net = comparsion_nn.Hybird_net(Bands, class_count, patch_size=patch_size)
# print(net)
# net.to(device)
# torch.cuda.empty_cache()
# net = comparsion_nn.train_model(net, train_loader, val_loader, 5e-4, train_label_weight.astype('float32'), device, num_epochs=100)
# OA, classification_map = comparsion_nn.test_model(net, test_loader, test_label_mask.astype('bool'),
#                                                   Height, Width, device,dataset_name, method)
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//HybirdCNN-" + dataset_name + "-" + str(OA))
# ### DBDAnet
# method = 'DBDAnet'
#
# HSI_n = np.reshape(data, [Height, Width, -1])
# patch_size = 9
# train_dataset = comparsion_nn.HSI2D_dataset(HSI_n, train_rand_idx, patch_size=patch_size,
#                                             flip_augmentation=True, radiation_augmentation=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataset = comparsion_nn.HSI2D_dataset(HSI_n, val_rand_idx, patch_size=patch_size)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
# test_dataset = comparsion_nn.HSI2D_dataset(HSI_n, np.concatenate((np.arange(Height*Width)[None],Groundtruth[None]), axis=0).T, patch_size=patch_size)
# test_loader = DataLoader(test_dataset, batch_size=64)
# net = comparsion_nn.DBDA_network_MISH(Bands, class_count)
# print(net)
# net.to(device)
# torch.cuda.empty_cache()
# net = comparsion_nn.train_model(net, train_loader, val_loader, 5e-4, train_label_weight.astype('float32'), device, num_epochs=100)
# OA, classification_map = comparsion_nn.test_model(net, test_loader, test_label_mask.astype('bool'),
#                                                   Height, Width, device, dataset_name, method)
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//DBDAnet-" + dataset_name + "-" + str(OA))

### Hybirdsn_net
# method = 'Hybirdsn_net'
# pca_bands = 30
# pca = PCA(n_components=pca_bands, svd_solver='randomized')
# data_pca = pca.fit_transform(data)
# HSI_n = np.reshape(data_pca, [Height, Width, -1])
# patch_size = 25
# train_dataset = comparsion_nn.HSI2D_dataset(HSI_n, train_rand_idx, patch_size=patch_size,
#                                             flip_augmentation=True, radiation_augmentation=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataset = comparsion_nn.HSI2D_dataset(HSI_n, val_rand_idx, patch_size=patch_size)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
# test_dataset = comparsion_nn.HSI2D_dataset(HSI_n, np.concatenate((np.arange(Height*Width)[None], Groundtruth[None]),axis =0).T, patch_size=patch_size)
# test_loader = DataLoader(test_dataset, batch_size=64)
# net = comparsion_nn.Hybirdsn_net(pca_bands, class_count, patch_size=patch_size)
# print(net)
# net.to(device)
# torch.cuda.empty_cache()
# net = comparsion_nn.train_model(net, train_loader, val_loader, 5e-4, train_label_weight.astype('float32'), device, num_epochs=100)
# OA, classification_map = comparsion_nn.test_model(net, test_loader, test_label_mask.astype('bool'),
#                                                   Height, Width, device, dataset_name, method)
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//Hybirdsn_net-" + dataset_name + "-" + str(OA))
###SSRN_network
# method = 'SSRN_network'
# HSI_n = np.reshape(data, [Height, Width, -1])
# patch_size = 7
# train_dataset = comparsion_nn.HSI2D_dataset(HSI_n, train_rand_idx, patch_size=patch_size,
#                                             flip_augmentation=True, radiation_augmentation=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataset = comparsion_nn.HSI2D_dataset(HSI_n, val_rand_idx, patch_size=patch_size)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
# test_dataset = comparsion_nn.HSI2D_dataset(HSI_n, np.concatenate((np.arange(Height*Width)[None],Groundtruth[None]),axis =0).T, patch_size=patch_size)
# test_loader = DataLoader(test_dataset, batch_size=64)
# net = comparsion_nn.SSRN_network(Bands, class_count)
# print(net)
# net.to(device)
# torch.cuda.empty_cache()
# net = comparsion_nn.train_model(net, train_loader, val_loader, 5e-4, train_label_weight.astype('float32'), device, num_epochs=100)
# OA, classification_map = comparsion_nn.test_model(net, test_loader, test_label_mask.astype('bool'),
#                                                   Height, Width, device,dataset_name, method)
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//SSRN_network-" + dataset_name + "-" + str(OA))
# ##FDSSC_network
# method = 'FDSSC_network'
# HSI_n = np.reshape(data, [Height, Width, -1])
# patch_size = 9
# train_dataset = comparsion_nn.HSI2D_dataset(HSI_n, train_rand_idx, patch_size=patch_size,
#                                             flip_augmentation=True, radiation_augmentation=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_dataset = comparsion_nn.HSI2D_dataset(HSI_n, val_rand_idx, patch_size=patch_size)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
# test_dataset = comparsion_nn.HSI2D_dataset(HSI_n, np.concatenate((np.arange(Height*Width)[None],Groundtruth[None]),axis =0).T, patch_size=patch_size)
# test_loader = DataLoader(test_dataset, batch_size=64)
# net = comparsion_nn.FDSSC_network(Bands, class_count)
# print(net)
# net.to(device)
# torch.cuda.empty_cache()
# net = comparsion_nn.train_model(net, train_loader, val_loader, 5e-4, train_label_weight.astype('float32'), device, num_epochs=100)
# OA, classification_map = comparsion_nn.test_model(net, test_loader, test_label_mask.astype('bool'),
#                                                   Height, Width, device,dataset_name, method)
# HSI_SELF.Draw_Classification_Map(classification_map+1, "results//FDSSC_network-" + dataset_name + "-" + str(OA))
###############GNN
self_data = HSI_SELF.SELF(data, train_rand_idx, 7, self_Thres, 1.0)  # SELF 提取特征
self_data_Bands = self_data.shape[1]
HSI_SELF.image_show(self_data, Height, Width)
edge_index_list, Qmatrix_list = HSI_SELF.SLIC_seg(HSI[:, :, RGB_bands],
                                                  np.reshape(self_data, [Height, Width, -1]), segs_list)

# load data in GPU

train_label_mask = torch.from_numpy(train_label_mask.astype(bool)).to(device)
test_label_mask = torch.from_numpy(test_label_mask.astype(bool)).to(device)
val_label_mask = torch.from_numpy(val_label_mask.astype(bool)).to(device)
train_label_weight = torch.from_numpy(train_label_weight.astype(np.float32)).to(device)
Y_label = torch.from_numpy(Y.astype(np.int64)).to(device)
HSI_n = np.reshape(self_data, [Height, Width, -1])
net_input = np.array(HSI_n, np.float32)
net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)
edge_index_list_GPU = []
Qmatrix_list_GPU = []
Q_Hat_T_list_GPU = []
for i in range(len(edge_index_list)):
    edge_index_list_GPU.append(torch.from_numpy(edge_index_list[i].astype(np.int64)).to(device))
    Qmatrix_list_GPU.append(torch.from_numpy(Qmatrix_list[i]).to_sparse_coo().to(device))
    Q_Hat_T_list_GPU.append(
        torch.from_numpy((Qmatrix_list[i] / (np.sum(Qmatrix_list[i], axis=0, keepdims=True))).T).to_sparse_coo().to(
            device))


def evaluate_performance(network_output, label, method=None, require_AA_KPP=False, printFlag=True):
    if not require_AA_KPP:
        with torch.no_grad():
            _, pred = network_output.max(dim=1)
            correct = float(pred.eq(label).sum().item())
            OA = correct / (label != 255).float().sum()
            return OA
    else:
        with torch.no_grad():
            # OA
            _, pred = network_output.max(dim=1)
            correct = float(pred.eq(label).sum().item())
            OA = correct / (label != 255).float().sum()
            OA = OA.cpu().numpy()
            y_pred = pred.cpu().numpy()
            gt_label = label.cpu().numpy()
            # Kappa
            kappa = metrics.cohen_kappa_score(y_pred, gt_label)
            # confusion_matrix
            confusion_matrix = metrics.confusion_matrix(gt_label, y_pred)
            producer_acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
            AA = np.average(producer_acc)

            print("test OA=", OA, 'test AA=', AA, 'kpp=', kappa, 'confusion matrix=', confusion_matrix)
            # save
            f = open('results//' + method + '-' + dataset_name + '_results.txt', 'a+')
            str_results = '\n======================' \
                          + " learning rate=" + str(learning_rate) \
                          + " epochs=" + str(max_epoch) \
                          + " train ratio=" + str(train_ratio) \
                          + " val ratio=" + str(val_ratio) \
                          + " ======================" \
                          + "\nOA=" + str(OA) \
                          + "\nAA=" + str(AA) \
                          + '\nkpp=' + str(kappa) \
                          + '\nacc per class:' + str(producer_acc) \
                          + "\nconfusion matrix:" + str(confusion_matrix) + "\n"

            f.write(str_results)
            f.close()
            return OA


##MSDesGATnet
net = HSI_SELF.MSDesGATnet(Height, Width, self_data_Bands, class_count, Qmatrix_list_GPU,
                           Q_Hat_T_list_GPU, edge_index_list_GPU,
                           SSConv_num_depth_conv_layer=SSConv_num_depth_conv_layer,
                           SSConv_kernel=SSConv_kernel)

print("parameters", net.parameters(), len(list(net.parameters())))
print(net)
net.to(device)



optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
best_loss = 1e10
best_OA = 0
torch.cuda.empty_cache()
net.train()
#net.load_state_dict(torch.load("model\\best_model.pt"))

for epochs in range(max_epoch + 1):
    optimizer.zero_grad()
    output = net(net_input)
    loss = F.cross_entropy(output[train_label_mask], Y_label[train_label_mask], train_label_weight)
    loss.backward()
    optimizer.step()
    if epochs % 10 == 0:
        with torch.no_grad():
            net.eval()
            output = net(net_input)
            trainloss = F.cross_entropy(output[train_label_mask], Y_label[train_label_mask], train_label_weight)
            trainOA = evaluate_performance(output[train_label_mask], Y_label[train_label_mask])
            valloss = F.cross_entropy(output[val_label_mask], Y_label[val_label_mask], train_label_weight)
            valOA = evaluate_performance(output[val_label_mask], Y_label[val_label_mask])
            print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(epochs + 1), trainloss, trainOA,
                                                                                   valloss, valOA))

            if valOA > best_OA or best_loss > valloss:
                best_loss = valloss
                best_OA = valOA
                torch.save(net.state_dict(), "model//best_model.pt")
                print('save model...')

        torch.cuda.empty_cache()
        net.train()

print("\n\n====================training done. starting evaluation...========================\n")
torch.cuda.empty_cache()


with torch.no_grad():
    net.load_state_dict(torch.load("model//best_model.pt"))
    net.eval()
    output = net(net_input)
    testloss = F.cross_entropy(output[test_label_mask], Y_label[test_label_mask], train_label_weight)
    testOA = evaluate_performance(output[test_label_mask], Y_label[test_label_mask], method='MSDesGATnet',
                                  require_AA_KPP=True)
    classification_map = torch.argmax(output, 1).reshape([Height, Width]).cpu() + 1
    HSI_SELF.Draw_Classification_Map(classification_map, "results//MSDesGATnet-" + dataset_name + str(testOA))
torch.cuda.empty_cache()
del net
##EGNN
# net = HSI_SELF.EGNN(Height, Width, self_data_Bands, class_count, 64, Qmatrix_list_GPU[0], Q_Hat_T_list_GPU[0],
#                     edge_index_list_GPU[0])
# print("parameters", net.parameters(), len(list(net.parameters())))
# print(net)
# net.to(device)
#
# learning_rate = 1e-3
#
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
# best_loss = 1e10
# best_OA = 0
# torch.cuda.empty_cache()
# net.train()
# for epochs in range(max_epoch + 1):
#     optimizer.zero_grad()
#     output = net(net_input)
#     loss = F.cross_entropy(output[train_label_mask], Y_label[train_label_mask], train_label_weight)
#     loss.backward()
#     optimizer.step()
#     if epochs % 10 == 0:
#         with torch.no_grad():
#             net.eval()
#             output = net(net_input)
#             trainloss = F.cross_entropy(output[train_label_mask], Y_label[train_label_mask], train_label_weight)
#             trainOA = evaluate_performance(output[train_label_mask], Y_label[train_label_mask])
#             valloss = F.cross_entropy(output[val_label_mask], Y_label[val_label_mask], train_label_weight)
#             valOA = evaluate_performance(output[val_label_mask], Y_label[val_label_mask])
#             print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(epochs + 1), trainloss, trainOA,
#                                                                                    valloss, valOA))
#
#             if valOA > best_OA:
#                 best_loss = valloss
#                 best_OA = valOA
#                 torch.save(net.state_dict(), "model//best_model.pt")
#                 print('save model...')
#
#         torch.cuda.empty_cache()
#         net.train()
#
# print("\n\n====================training done. starting evaluation...========================\n")
# torch.cuda.empty_cache()
# with torch.no_grad():
#     net.load_state_dict(torch.load("model//best_model.pt"))
#     net.eval()
#     output = net(net_input)
#     testloss = F.cross_entropy(output[test_label_mask], Y_label[test_label_mask], train_label_weight)
#     testOA = evaluate_performance(output[test_label_mask], Y_label[test_label_mask], method='EGNN', require_AA_KPP=True)
#     classification_map = torch.argmax(output, 1).reshape([Height, Width]).cpu() + 1
#     HSI_SELF.Draw_Classification_Map(classification_map, "results//EGNN-" + dataset_name + str(testOA))
# torch.cuda.empty_cache()
# del net
