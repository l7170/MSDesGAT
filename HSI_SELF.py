import numpy as np
from scipy.linalg import eigh
from sklearn import preprocessing
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, EdgeConv
import spectral as spy
import os

def SELF(HSI, train_idx, K: int, eigs_thres: float , beta: float):
    num_total, bands = HSI.shape
    train_set = HSI[train_idx[:, 0], :]
    num_train = train_set.shape[0]
    total_2 = np.linalg.norm(HSI, axis=1)**2
    train_2 = np.linalg.norm(train_set, axis=1)**2
    total_sum = np.matmul(train_set, HSI.T)
    total_sum = np.tile(total_2[None],(num_train,1)) + np.tile(train_2[None].T,(1,num_total)) -2.0*total_sum
    total_sum = np.sqrt(np.where(total_sum>=0, total_sum,0))
    _ind = np.argsort(total_sum, axis=1)
    sigma_train = np.zeros(num_train)
    for i in range(num_train):
        sigma_train[i] = total_sum[i, _ind[i, K]]
    num_class= int(np.amax(train_idx[:, 1]))
    num_perclass = np.zeros(num_class,dtype='int')
    for i in range(num_class):
        idx = np.where(train_idx[:, 1]==i+1)[-1]
        num_perclass[i] = len(idx)
    W_lb = np.zeros((num_train,num_train))
    W_lw = np.zeros((num_train,num_train))
    for i in range(num_train):
        p_cur = train_set[i, :]
        p_lab = train_idx[i][1]
        for j in range(i, num_train):
            if train_idx[j][1] == p_lab:
                tmp1 = (1.0/num_train-1.0/num_perclass[p_lab-1])*\
                      np.exp(-np.linalg.norm(p_cur- train_set[j, :])**2/(sigma_train[i]*sigma_train[j]))
                tmp2 = (1.0/num_perclass[p_lab-1])*np.exp(-np.linalg.norm(p_cur- train_set[j, :])**2
                                                           /(sigma_train[i]*sigma_train[j]))
            else:
                tmp1 = 1.0/num_train
                tmp2 = 0.0
            W_lb[i, j] = W_lb[j,i] = tmp1
            W_lw[i, j] = W_lw[j,i] = tmp2

    S_lb = np.matmul(np.matmul(train_set.T,np.diag(np.sum(W_lb, axis=1))-W_lb),train_set)
    S_lw = np.matmul(np.matmul(train_set.T,np.diag(np.sum(W_lw, axis=1))-W_lw),train_set)
    mu = (np.sum(HSI, axis=0)/num_total)[:, None]
    S_t = np.matmul(HSI.T, HSI)- np.matmul(mu,mu.T)*num_total
    S_rlb = (1.0-beta)*S_lb + beta*S_t
    S_rlw = (1.0-beta)*S_lw + beta*np.eye(bands)
    eigs_val, vec_eigs = eigh(S_rlb, S_rlw)

    compress_num = int(np.min(np.where(np.flip(eigs_val).cumsum(0) / np.sum(eigs_val) > eigs_thres))+1)

    T = np.flip(vec_eigs[:, -compress_num:], axis=1)
    #T = np.multiply(T, np.tile(np.sqrt(np.flip(eigs_val[-compress_num:])[None]), (bands,1)))
    return np.matmul(HSI, T)

def image_show(data, Height, Width):
    minMax = preprocessing.MinMaxScaler()
    img = minMax.fit_transform(data[:, :3])
    img = np.reshape(img, [Height, Width, 3])
    plt.imshow(img)
    plt.pause(1)

def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.pause(1)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass

def SegmentsLabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels

def SLIC_seg(HSI, Datacube, segments_list):
    H, W, bands =Datacube.shape

    minMax2 = preprocessing.MinMaxScaler()
    HSI = np.reshape(HSI,[H*W,3])
    HSI = minMax2.fit_transform(HSI)
    HSI = np.reshape(HSI,[H,W,3])
    num_scale = len(segments_list)
    node_feature_list = []
    edge_index_list = []
    Qmatrix_list = []
    for k in range(num_scale):
        segments = slic(Datacube, n_segments=segments_list[k], compactness=0.1)
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)
        show_seg = mark_boundaries(HSI, segments)
        plt.figure()
        plt.imshow(show_seg)
        plt.show()

        superpixel_num_cur = segments.max() + 1
        segments = np.reshape(segments, [-1])
        Q = np.zeros([H * W, superpixel_num_cur], dtype=np.float32)

        for i in range(superpixel_num_cur):
            idx = np.where(segments == i)[0]
            Q[idx, i] = 1
        segments = np.reshape(segments, [H, W])
        Affine=np.zeros((superpixel_num_cur,superpixel_num_cur), dtype='int')
        for i in range(H):
            for j in range(W - 1):
                if (segments[i, j] != segments[i, j + 1]):
                    Affine[segments[i, j], segments[i, j + 1]] = Affine[segments[i, j + 1], segments[i, j]] = 1

        for i in range(H - 1):
            for j in range(W):
                if (segments[i, j] != segments[i + 1, j]):
                    Affine[segments[i, j], segments[i + 1, j]] = Affine[segments[i + 1, j], segments[i, j]] = 1

        edge_index = np.empty((2, 0), dtype=int)
        for i in range(superpixel_num_cur):
            for j in range(i + 1, superpixel_num_cur):
                if (Affine[i, j] != 0):
                    edge_index = np.concatenate((edge_index, np.array([[i, j], [j, i]])), axis=1)

        #node_feature_list.append(S)
        edge_index_list.append(edge_index)
        Qmatrix_list.append(Q)
    return edge_index_list, Qmatrix_list


class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch, num_depth_conv_layer, kernel_size=5):
        super(SSConv, self).__init__()
        self.num_depth_conv_layer = num_depth_conv_layer
        self.depth_conv = nn.Sequential()
        for i in range(self.num_depth_conv_layer):
            self.depth_conv.add_module('depth_conv_'+str(i),nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                                    kernel_size = kernel_size, stride=1, padding=kernel_size//2, groups=out_ch))

        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )
        self.Act = nn.LeakyReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_ch)


    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act(out)
        for i in range(self.num_depth_conv_layer):
            out = self.depth_conv[i](out)
            out = self.Act(out)
        return out

class myGNN(torch.nn.Module):
    def __init__(self, num_inputfeatures: int, num_outfeatures: int):
        super().__init__()
        self.BN = nn.BatchNorm1d(num_inputfeatures)
        self.conv1 = GATv2Conv(num_inputfeatures, 64, heads=4)
        self.conv2 = GATv2Conv(256, num_outfeatures, heads=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.BN(x)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        return x


class MSDesGATnet(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_count: int,
                 Qmatrix_list, Q_Hat_T_list, edge_index_list, SSConv_num_depth_conv_layer = 1,SSConv_kernel = 5):
        super(MSDesGATnet, self).__init__()
        self.class_count = class_count
        self.channel = channel
        self.height = height
        self.width = width
        self.edge_index_list = edge_index_list
        self.Qmatrix_list = Qmatrix_list
        self.layer_num = len(Qmatrix_list)
        self.Q_list_Hat_T = Q_Hat_T_list
        self.GAT_layers = nn.Sequential()
        input_num = channel
        output_num = channel
        for i in range(self.layer_num):
            self.GAT_layers.add_module('my_GNN_l'+str(i), myGNN(input_num, output_num))
            input_num = input_num + output_num
            output_num = input_num

        self._linear = nn.Linear(input_num, self.class_count)
        self._CNN_denoise1 = SSConv(channel, channel,num_depth_conv_layer=SSConv_num_depth_conv_layer,kernel_size=SSConv_kernel)
        self._CNN_denoise2 = SSConv(output_num, output_num,num_depth_conv_layer=SSConv_num_depth_conv_layer,kernel_size=SSConv_kernel)

    def forward(self, x: torch.Tensor):
        (h, w, c) = x.shape
        #x0 = x.reshape([h * w, -1])
        x = torch.unsqueeze(x.permute([2, 0, 1]), 0)
        H_0 = self._CNN_denoise1(x)
        H_0 = torch.squeeze(H_0, 0).permute([1, 2, 0])
        x_flatten = H_0.reshape([h * w, -1])

        for i in range(self.layer_num):
            superpixels_flatten = torch.sparse.mm(self.Q_list_Hat_T[i], x_flatten)
            H_i = self.GAT_layers[i](superpixels_flatten, self.edge_index_list[i])
            x_flatten = torch.cat([x_flatten, torch.sparse.mm(self.Qmatrix_list[i], H_i)], dim=-1)

        output = self._CNN_denoise2(x_flatten.reshape([1,h,w,-1]).permute([0,3,1,2]))
        x_flatten = torch.squeeze(output, 0).permute([1, 2, 0]).reshape([h * w, -1])
        Y = self._linear(x_flatten)
        return Y

class EGNN(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_count: int, out_channel: int,
                 Qmatrix, Q_Hat_T, edge_index):
        super(EGNN, self).__init__()
        self.class_count = class_count
        self.channel = channel
        self.height = height
        self.width = width
        self.edge_index = edge_index
        self.Qmatrix = Qmatrix
        self.Q_list_Hat_T = Q_Hat_T
        self.GNN1 = EdgeConv(nn.Sequential(nn.Linear(2*channel, out_channel), nn.ReLU(inplace=True)))
        self.GNN2 = EdgeConv(nn.Sequential(nn.Linear(2*out_channel, out_channel), nn.ReLU(inplace=True)))
        self.GNN3 = EdgeConv(nn.Sequential(nn.Linear(2*out_channel, out_channel), nn.ReLU(inplace=True)))
        self.CNN = nn.Conv2d(out_channel, out_channel,kernel_size=3,stride=1,padding=1,groups=1)
        self._linear = nn.Linear(out_channel, self.class_count)

    def forward(self, x: torch.Tensor):
        (h, w, c) = x.shape
        x_flatten = x.reshape([h * w, -1])
        superpixels_flatten = torch.sparse.mm(self.Q_list_Hat_T, x_flatten)
        H1 = self.GNN1(superpixels_flatten, self.edge_index)
        H2 = self.GNN2(H1, self.edge_index)
        H3 = self.GNN2(H2, self.edge_index)
        H4 = H1 + H2 + H3
        output = torch.sparse.mm(self.Qmatrix, H4).reshape([1,h,w,-1]).permute([0,3,1,2])
        output = self.CNN(output)
        output = torch.squeeze(output, 0).permute([1, 2, 0]).reshape([h * w, -1])
        output =self._linear(output)
        return output

def write_acc(OA, AA, producer_acc, kappa, confusion_matrix, filename, method):
    if os.path.exists(filename):
        f = open(filename, 'a+')
    else:
        f = open(filename, 'w')
    str_results = '\n' + method + ': ======================' \
                  + "\nOA=" + str(OA) \
                  + "\nAA=" + str(AA) \
                  + '\nkpp=' + str(kappa) \
                  + '\nacc per class:' + str(producer_acc) \
                  + "\nconfusion matrix:" + str(confusion_matrix) + "\n"

    f.write(str_results)
    f.close()





