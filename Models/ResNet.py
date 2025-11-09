import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
from d2l import torch as d2l
from torch.nn import functional as F
import NxUtil
import numpy as np

# 获取数据
train_img, train_lable = NxUtil.ImageDataLoading("./Data/Train/", "./Data/train.csv", (224, 224), "image", "label")
# test_img = NxUtil.ValImageDataLoading("./Data/Test/", "./Data/test.csv", (224, 224), "image")

# 转换成Tensor格式
train_img = torch.tensor(train_img, dtype=torch.float32)
train_img = train_img.permute(0, 3, 1, 2)
# test_img = torch.tensor(test_img, dtype=torch.float32)
# test_img = test_img.permute(0, 3, 1, 2)

# print(train_lable)


# print(train_img.shape)
# print(test_img.shape)
# torch.Size([18353, 3, 224, 224])
# torch.Size([8800, 3, 224, 224])


def load_dataset(data_arry, batch_size, is_train=True):
    """
    构造数据迭代器
    :param data_arry: 数据
    :param batch_size: 批量大小
    :param is_train: 是否训练
    :return: Data.DataLoader  数据加载器
    """
    dataSet = data.TensorDataset(*data_arry)
    return data.DataLoader(dataSet, batch_size, shuffle=is_train)


train_iter = load_dataset((train_img, train_lable), 64)


# 块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 4))
b4 = nn.Sequential(*resnet_block(128, 256, 6))
b5 = nn.Sequential(*resnet_block(256, 512, 3))


net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 176))


# X = torch.rand(size=(1, 3, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape:\t', X.shape)

lr, num_epochs= 0.05, 10



def train(net, train_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    global metric, train_l, train_acc

    A = d2l.Timer()
    A.start()

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        print(f"第{epoch + 1}次训练误差为：{train_l:.3f}")
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}')
    d2l.plt.show()
    A.stop()
    print(f"用时：{A.sum()}")
