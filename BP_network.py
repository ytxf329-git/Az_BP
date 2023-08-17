# BP模块 借助PyTorch实现

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


# 引入了遗传算法参数的BP模型
class GABP_net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output, GA_parameter):
        super(GABP_net, self).__init__()
        # 构造隐含层和输出层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)
        # 给定网络训练的初始权值和偏执等
        self.hidden.weight = torch.nn.Parameter(GA_parameter[0])
        self.hidden.bias = torch.nn.Parameter(GA_parameter[1])
        self.output.weight = torch.nn.Parameter(GA_parameter[2])
        self.output.bias = torch.nn.Parameter(GA_parameter[3])

    def forward(self, x):
        # 前向计算
        hid = torch.sigmoid(self.hidden(x))
        out = torch.sigmoid(self.output(hid))
        return out


# 传统的BP模型
class ini_BP_net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(ini_BP_net, self).__init__()
        # 构造隐含层和输出层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 前向计算
        hid = torch.sigmoid(self.hidden(x))
        out = torch.sigmoid(self.output(hid))
        return out


def train(model, epochs, learning_rate, x_train, y_train):
    """
    :param model: 模型
    :param epochs: 最大迭代次数
    :param learning_rate:学习率
    :param x_train:训练数据（输入）
    :param y_train:训练数据（输出）
    :return: 最终的loss值（MSE）
    """
    # path = "log.txt"
    # f = open(path, 'w',encoding='UTF-8')
    # f.write("train log\n------Train Action------\n"
    #         "Time：{}\n".format(time.ctime()))
    loss_fc = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    loss_list = []
    for i in range(epochs):
        model.train()
        # 前向计算
        data = model(x_train)
        # 计算误差
        loss = loss_fc(data, y_train)
        loss_list.append(loss)
        # 更新梯度
        optimizer.zero_grad()
        # 方向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # print("This is {} th iteration,MSE is {}。".format(i+1,loss))
    loss_ls = [loss_list[i].detach().numpy() for i in range(len(loss_list))]
    return loss_ls


# BP网络调试模块
if __name__ == "__main__":
    n_feature = 2
    n_hidden = 5
    n_output = 1
    # 构建模型
    model = ini_BP_net(n_feature, n_hidden, n_output)
    # 测试
    # 数据准备
    # ======================== #
    x_train = torch.rand(5,n_feature)
    y_label = torch.rand(5,n_output)
    # ============================== #
    learn_date = 1e-2
    loss_ls = train(model, 1000, learn_date, x_train, y_label)
    plt.plot(loss_ls)
    plt.show()

