import BP_network
# 引入数据分析包
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
import torch
import time
from torchvision.transforms import transforms

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
start_time = time.time()

# 读取数据
df = pd.read_excel('10.xlsx')
env_T = df['环境温度']
hum_T = df['环境湿度']
wind_v = df['风速（机械）']
eff = df['有功功率限制值']

# 数据预处理，每n个取一个平均值
# env_T_data = []
# hum_T_data = []
# wind_v_data = []
# eff_data = []
# n = 5
# for i in range(0, len(env_T), n):
#     try:
#         # 温度数据
#         env_temp = round(sum([env_T[i + j] for j in range(n)]) / n, 3)
#         env_T_data.append(env_temp)
#         # 湿度数据
#         hum_temp = round(sum([hum_T[i + j] for j in range(n)]) / n, 3)
#         hum_T_data.append(hum_temp)
#         # 风速数据
#         wind_v_temp = round(sum([wind_v[i + j] for j in range(n)]) / n, 3)
#         wind_v_data.append(wind_v_temp)
#         # 功率数据
#         eff_temp = round(sum([eff[i + j] for j in range(n)]) / n, 3)
#         eff_data.append(eff_temp)
#     except:
#         pass

# 每5个点平均，数据做平滑处理
# 分割数据集
M = len(eff)
# np.array.T 用来转置矩阵
data_set = np.array([env_T, hum_T, wind_v, eff]).T
train_set = data_set[:int(M * 0.6), :]
# cv_set = data_set[int(M * 0.6):int(M * 0.8), :]
test_set = data_set[int(M * 0.8):, :]
print("训练集维度：", train_set.shape)
print("测试集维度：", test_set.shape)

# 首先对数据进行归一化处理
scale = MinMaxScaler()
train_set_norm = scale.fit_transform(train_set)
# cv_set_norm = scale.fit_transform(cv_set)
test_set_norm = scale.fit_transform(test_set)
print('数据准备完成...')

# 首先使用经典BP神经网络
# =========================== #
print("开始经典BP训练...")
n_feature = 3
n_hidden = 10
n_output = 1
num_epoch = 1500
learn_rate = 1e-3
BP_net = BP_network.ini_BP_net(n_feature, n_hidden, n_output)
x_train = train_set_norm[:, :3]
y_train = train_set_norm[:, 3].reshape(train_set_norm.shape[0], 1)
# print(x_train.shape,y_train.shape)
# 格式转换
tensor_tran = transforms.ToTensor()
x_train = tensor_tran(x_train).to(torch.float).reshape(x_train.shape[0], 3)
y_train = tensor_tran(y_train).to(torch.float).reshape(y_train.shape[0], 1)
# 进行训练
BP_lossList = BP_network.train(BP_net, num_epoch, learn_rate, x_train, y_train)
# 测试集
x_test = tensor_tran(test_set_norm[:, :3]).to(torch.float).reshape(test_set_norm[:, :3].shape[0], 3)
y_test = test_set_norm[:, 3].reshape(test_set_norm.shape[0], 1)
# 预测结果
BP_prediction = BP_net(x_test).detach().numpy()
print("经典BP训练完成...")
# 预测绘图

plt.figure()
plt.plot(BP_lossList, c='b')
plt.ylabel("BP误差下降曲线")


f1 = plt.figure()
plt.plot(BP_prediction)
plt.plot(y_test)
plt.show()