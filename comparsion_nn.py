import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import torch.optim as optim
from sklearn import metrics
import math
import matplotlib.pyplot as plt


class HSI1D_dataset(torch.utils.data.Dataset):
    def __init__(self, data, gt):
        super(HSI1D_dataset, self).__init__()
        num, bands = data.shape
        self.data = data
        self.label = gt
        self.indices = np.arange(num)
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = self.indices[i]
        data = self.data[x, :]
        label = self.label[x]
        data = torch.from_numpy(np.asarray(data.astype('float32')))
        label = torch.from_numpy(np.asarray(label.astype('int64')))
        return x, data, label


class HSI2D_dataset(torch.utils.data.Dataset):
    def __init__(self, data, label_idx, patch_size, flip_augmentation=False, radiation_augmentation=False):
        super(HSI2D_dataset, self).__init__()
        h, w, bands = data.shape
        p = patch_size // 2
        self.h = h
        self.w = w
        self.data = np.pad(data, ((p, p), (p, p), (0, 0)), mode='reflect')
        self.label = label_idx
        self.patch_size = patch_size
        self.indices = np.arange(label_idx.shape[0])
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(data):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            data = np.fliplr(data)
        if vertical:
            data = np.flipud(data)
        return data

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = self.indices[i]
        x_pos = int(self.label[x, 0] / self.w)
        y_pos = self.label[x, 0] % self.w
        data = self.data[x_pos:x_pos + self.patch_size, y_pos:y_pos + self.patch_size, :]
        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data = self.flip(data)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)

        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = self.label[x][1] - 1
        data = torch.from_numpy(np.asarray(data.astype('float32')))
        label = torch.from_numpy(np.asarray(label.astype('int64')))
        data = data.unsqueeze(0)
        return x, data, label


class Baseline(nn.Module):
    """
    Baseline network
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x


def train_model(model, traindataloader, valdataloader, learning_rate, class_weight, device, num_epochs=40):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weight).to(device))
    # criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        for _, (_, x, y) in enumerate(traindataloader):
            x, y = x.to(device), y.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_corrects += torch.sum(torch.argmax(output, 1) == y.data)
            train_num += x.size(0)
            del (x, y, loss, output)
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        print('{} Train Loss: {} Train OA: {}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        # scheduler.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                for _, (_, x, y) in enumerate(valdataloader):
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    loss = criterion(output, y)
                    val_loss += loss.item() * x.size(0)
                    val_corrects += torch.sum(torch.argmax(output, 1) == y.data)
                    val_num += x.size(0)

                val_loss_all.append(val_loss / val_num)
                val_acc_all.append(val_corrects.double().item() / val_num)
                print('{} val Loss: {} val OA: {}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
                if val_acc_all[-1] > best_acc:
                    best_acc = val_acc_all[-1]
                    torch.save(model.state_dict(), "model//best_model.pt")
                    print('save model...')
    print('=' * 60)
    return model


def test_model(model, testdataloader, mask, Height, Width, device, dataset_name, method):
    model.load_state_dict(torch.load("model//best_model.pt"))
    model.eval()
    test_y_all = torch.LongTensor().to(device)
    pre_y_all = torch.LongTensor().to(device)
    ind_all = torch.LongTensor().to(device)
    for _, (ind, x, y) in enumerate(testdataloader):
        ind, x, y = ind.to(device), x.to(device), y.to(device)
        out = model(x)
        pre_lab = torch.argmax(out, 1)
        test_y_all = torch.cat((test_y_all, y))
        pre_y_all = torch.cat((pre_y_all, pre_lab))
        ind_all = torch.cat((ind_all, ind))

    test_y_all = test_y_all.cpu()
    pre_y_all = pre_y_all.cpu()
    ind_all = ind_all.cpu()
    y_map = np.zeros(Height * Width, dtype='int')
    pre_map = np.zeros(Height * Width, dtype='int')
    y_map[ind_all] = test_y_all
    pre_map[ind_all] = pre_y_all

    OA = metrics.accuracy_score(y_map[mask], pre_map[mask])
    kappa = metrics.cohen_kappa_score(pre_map[mask], y_map[mask])
    # confusion_matrix
    confusion_matrix = metrics.confusion_matrix(y_map[mask], pre_map[mask])
    producer_acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    AA = np.average(producer_acc)

    print("\ntest OA=", OA, ' test AA=', AA, ' kpp=', kappa, "\nproducer_acc",
          producer_acc, '\nconfusion matrix=', confusion_matrix)
    # save
    f = open('results//' + dataset_name + '_results.txt', 'a+')
    str_results = '\n' + method + ': ======================' \
                  + "\nOA=" + str(OA) \
                  + "\nAA=" + str(AA) \
                  + '\nkpp=' + str(kappa) \
                  + '\nacc per class:' + str(producer_acc) \
                  + "\nconfusion matrix:" + str(confusion_matrix) + "\n"

    f.write(str_results)
    f.close()
    classification_map = np.reshape(pre_map, [Height, Width])
    return OA, classification_map


def map_classification(model, data_img, Height: int, Width: int):
    model.load_state_dict(torch.load("model//best_model.pt"))
    model.eval()


class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            n2 = input_channels - kernel_size + 1
            pool_size = math.ceil(n2 / 40)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class MouEtAl(nn.Module):
    """
    Deep recurrent neural networks for hyperspectral image classification
    Lichao Mou, Pedram Ghamisi, Xiao Xang Zhu
    https://ieeexplore.ieee.org/document/7914752/
    """

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform_(m.weight.data, -0.1, 0.1)
            init.uniform_(m.bias.data, -0.1, 0.1)

    def __init__(self, input_channels, n_classes):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(MouEtAl, self).__init__()
        self.input_channels = input_channels
        self.gru = nn.GRU(1, 64, 1, bidirectional=False)  # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d(64 * input_channels)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(64 * input_channels, n_classes)

    def forward(self, x):
        x = x.squeeze()
        x = x.unsqueeze(0)
        # x is in 1, N, C but we expect C, N, 1 for GRU layer
        x = x.permute(2, 1, 0)
        x = self.gru(x)[0]
        # x is in C, N, 64, we permute back
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(x.size(0), -1)
        x = self.gru_bn(x)
        x = self.tanh(x)
        x = self.fc(x)
        return x


class HeEtAl(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)
        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)
        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               (3, 3, 3), padding=(1, 0, 0))
        # self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        out = self.fc(x)
        return out


class Hybird_net(nn.Module):
    def __init__(self, input_channels, n_classes, patch_size=11):
        super(Hybird_net, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.conv1d_1 = nn.Sequential(nn.Conv1d(input_channels, 300, kernel_size=3), nn.MaxPool1d(2, 2), nn.ReLU())
        self.conv1d_2 = nn.Sequential(nn.Conv1d(300, 200, kernel_size=3), nn.MaxPool1d(2, 2), nn.ReLU())

        self.conv2d_1 = nn.Sequential(nn.Conv2d(input_channels, 300, kernel_size=(3, 3)), nn.MaxPool2d((2, 2), 2),
                                      nn.ReLU())
        self.conv2d_2 = nn.Sequential(nn.Conv2d(300, 200, kernel_size=(3, 3)), nn.MaxPool2d((2, 2), 2), nn.ReLU())

        self.conv3d_1 = nn.Sequential(nn.Conv3d(1, 2, kernel_size=(7, 3, 3)), nn.MaxPool3d((5, 5, 5), 2), nn.ReLU())
        self.conv3d_2 = nn.Sequential(nn.Conv3d(2, 4, kernel_size=(7, 3, 3)), nn.ReLU())

        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x_1d = x.reshape([-1, self.input_channels, self.patch_size * self.patch_size])
            x_1d = self.conv1d_1(x_1d)
            x_1d = self.conv1d_2(x_1d)
            x_1d = torch.flatten(x_1d, start_dim=1)

            x_2d = self.conv2d_1(x.squeeze(dim=1))
            x_2d = self.conv2d_2(x_2d)
            x_2d = torch.flatten(x_2d, start_dim=1)

            x_3d = self.conv3d_1(x)
            x_3d = self.conv3d_2(x_3d)
            x_3d = torch.flatten(x_3d, start_dim=1)
        return x_1d.size(1) + x_2d.size(1) + x_3d.size(1)

    def forward(self, x):
        x_1d = torch.reshape(x, [-1, self.input_channels, self.patch_size * self.patch_size])
        x_1d = self.conv1d_1(x_1d)
        x_1d = self.conv1d_2(x_1d)
        x_1d = torch.flatten(x_1d, start_dim=1)

        x_2d = self.conv2d_1(x.squeeze(dim=1))
        x_2d = self.conv2d_2(x_2d)
        x_2d = torch.flatten(x_2d, start_dim=1)

        x_3d = self.conv3d_1(x)
        x_3d = self.conv3d_2(x_3d)
        x_3d = torch.flatten(x_3d, start_dim=1)
        out = self.fc(torch.cat((x_1d, x_2d, x_3d), dim=1))
        return out


class DCNN(nn.Module):
    def __init__(self, input_channels, n_classes, patch_size=5):
        super(DCNN, self).__init__()
        self.input_channels = input_channels
        self.n_cls = n_classes
        self.patch_size = patch_size

        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 3 * input_channels, kernel_size=3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(3 * input_channels, 9 * input_channels, kernel_size=3), nn.ReLU())

        self.mlp = nn.Sequential(nn.Linear(9 * input_channels, 6 * input_channels),
                                 nn.Linear(6 * input_channels, n_classes))

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.squeeze(x)
        out = self.mlp(x)
        return out


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # m_batchsize, channel, height, width, C = x.size()
        x = x.squeeze(-1)
        # m_batchsize, C, height, width, channel = x.size()

        # proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*channle).permute(0, 2, 1)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*channle)
        # energy = torch.bmm(proj_query, proj_key)
        # attention = self.softmax(energy)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*channle)
        #
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = (self.gamma * out + x).unsqueeze(-1)
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, channel = x.size()
        # print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # 形状转换并交换维度
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channel)
        # print('out', out.shape)
        # print('x', x.shape)

        out = self.gamma * out + x  # C*H*W
        return out


class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()

    # Also see https://arxiv.org/abs/1606.08415

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class DBDA_network_MISH(nn.Module):
    def __init__(self, band, classes):
        super(DBDA_network_MISH, self).__init__()

        # spectral branch
        self.name = 'DBDA_MISH'
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new()
            # swish()
            mish()
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv13 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=60, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))  # kernel size随数据变化

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        self.conv25 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                      kernel_size=(3, 3, 2), stride=(1, 1, 1)),
            nn.Sigmoid()
        )

        self.batch_norm_spectral = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new(),
            # swish(),
            mish(),
            nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new(),
            # swish(),
            mish(),
            nn.Dropout(p=0.5)
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(120, classes)  # ,
            # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

        # fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        batch_num, C, channel, H, W=X.shape
        X = X.permute(0, 1, 3, 4, 2)
        # spectral
        x11 = self.conv11(X)
        # print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        # print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        # print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        # print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        # print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        #x1 = torch.mul(x1, x16)

        # spatial
        # print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        # print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)

        # 空间注意力机制
        x2 = self.attention_spatial(x25)
        #x2 = torch.mul(x2, x25)

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.batch_norm_spatial(x2)
        x2 = self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        # print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output


class Hybirdsn_net(nn.Module):
    def __init__(self, input_channel: int, num_class: int, patch_size: int):
        super(Hybirdsn_net, self).__init__()
        self.input_channel = input_channel
        self.conv3d_1 = nn.Sequential(nn.Conv3d(1, 8, kernel_size=(3, 3, 7)), nn.ReLU())
        self.conv3d_2 = nn.Sequential(nn.Conv3d(8, 16, kernel_size=(3, 3, 5)), nn.ReLU())
        self.conv3d_3 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=(3, 3, 3)), nn.ReLU())
        self.conv2d_1 = nn.Sequential(nn.Conv2d((input_channel - 12) * 32, 64, kernel_size=(3, 3)), nn.ReLU())

        self.fc1 = nn.Sequential(nn.Linear((patch_size - 8) * (patch_size - 8) * 64, 256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, x):
        x = x.permute(0, 1, 3, 4, 2)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape([x.size(0), x.size(1), x.size(2), -1])
        x = x.permute(0, 3, 1, 2)
        x = self.conv2d_1(x)
        x = x.reshape([x.shape[0], -1])
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class SSRN_network(nn.Module):
    def __init__(self, band, classes):
        super(SSRN_network, self).__init__()
        self.name = 'SSRN'
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                               kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.res_net1 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net2 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(24, 24, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(24, 24, (3, 3, 1), (1, 1, 0))

        kernel_3d = math.ceil((band - 6) / 2)

        self.conv2 = nn.Conv3d(in_channels=24, out_channels=128, padding=(0, 0, 0),
                               kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=24, padding=(0, 0, 0),
                               kernel_size=(3, 3, 128), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(24, classes)  # ,
            # nn.Softmax()
        )

    def forward(self, X):
        X = X.permute(0, 1, 3, 4, 2)
        x1 = self.batch_norm1(self.conv1(X))
        # print('x1', x1.shape)

        x2 = self.res_net1(x1)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        # print(x10.shape)
        return self.full_connection(x4)


class FDSSC_network(nn.Module):
    def __init__(self, band, classes):
        super(FDSSC_network, self).__init__()

        # spectral branch
        self.name = 'FDSSC'
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                               kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.PReLU()
        )
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv3 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv4 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm4 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        kernel_3d = math.ceil((band - 6) / 2)
        # print(kernel_3d)
        self.conv5 = nn.Conv3d(in_channels=60, out_channels=200, padding=(0, 0, 0),
                               kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))

        self.batch_norm5 = nn.Sequential(
            nn.BatchNorm3d(1, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv6 = nn.Conv3d(in_channels=1, out_channels=24, padding=(1, 1, 0),
                               kernel_size=(3, 3, 200), stride=(1, 1, 1))
        self.batch_norm6 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.PReLU()
        )
        self.conv7 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm7 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv8 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm8 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv9 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm9 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(60, classes)
            # nn.Softmax()
        )

    def forward(self, X):
        X = X.permute(0, 1, 3, 4, 2)
        # spectral
        x1 = self.conv1(X)
        # print('x11', x11.shape)
        x2 = self.batch_norm1(x1)
        x2 = self.conv2(x2)
        # print('x12', x12.shape)

        x3 = torch.cat((x1, x2), dim=1)
        # print('x13', x13.shape)
        x3 = self.batch_norm2(x3)
        x3 = self.conv3(x3)
        # print('x13', x13.shape)

        x4 = torch.cat((x1, x2, x3), dim=1)
        x4 = self.batch_norm3(x4)
        x4 = self.conv4(x4)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        # print('x15', x15.shape)

        # print(x5.shape)
        x6 = self.batch_norm4(x5)
        x6 = self.conv5(x6)
        # print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        x6 = x6.permute(0, 4, 2, 3, 1)
        # print(x6.shape)

        x7 = self.batch_norm5(x6)
        x7 = self.conv6(x7)

        x8 = self.batch_norm6(x7)
        x8 = self.conv7(x8)

        x9 = torch.cat((x7, x8), dim=1)
        x9 = self.batch_norm7(x9)
        x9 = self.conv8(x9)

        x10 = torch.cat((x7, x8, x9), dim=1)
        x10 = self.batch_norm8(x10)
        x10 = self.conv9(x10)

        x10 = torch.cat((x7, x8, x9, x10), dim=1)
        x10 = self.batch_norm9(x10)
        x10 = self.global_pooling(x10)
        x10 = x10.view(x10.size(0), -1)

        output = self.full_connection(x10)
        # output = self.fc(x_pre)
        return output