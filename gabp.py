# 个人编写的遗传算法依赖
import chrom_code
import chrom_mutate
import chrom_cross
import chrom_select
import chrom_fitness
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
env_T_data = []
hum_T_data = []
wind_v_data = []
eff_data = []
n = 5
for i in range(0, len(env_T), n):
	try:
		# 温度数据
		env_temp = round(sum([env_T[i + j] for j in range(n)]) / n, 3)
		env_T_data.append(env_temp)
		# 湿度数据
		hum_temp = round(sum([hum_T[i + j] for j in range(n)]) / n, 3)
		hum_T_data.append(hum_temp)
		# 风速数据
		wind_v_temp = round(sum([wind_v[i + j] for j in range(n)]) / n, 3)
		wind_v_data.append(wind_v_temp)
		# 功率数据
		eff_temp = round(sum([eff[i + j] for j in range(n)]) / n, 3)
		eff_data.append(eff_temp)
	except:
		pass
# 每5个点平均，数据做平滑处理
# 分割数据集
M = len(env_T_data)
# np.array.T 用来转置矩阵
data_set = np.array([env_T_data, hum_T_data, wind_v_data, eff_data]).T
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
learn_rate = 1e-2
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
# ================================= #
# plt.figure()
# plt.plot(BP_prediction,label='BP prediction',c='b')
# plt.plot(y_test,label='True value',c='r')
# plt.grid(ls='--')
# plt.legend()
# plt.show()
# sys.exit()	# 程序运行到此处结束

# 基于遗传算法优化的BP神经网络 #
# ================================= #
print("开始进行遗传优化...")
chrom_len = n_feature * n_hidden + n_hidden + n_hidden * n_output + n_output  # 染色体长度
size = 15  # 种群规模,用来轮盘赌选择
bound = np.ones((chrom_len, 2))
sz = np.array([[-1, 0], [0, 1]])
bound = np.dot(bound, sz)  # 各基因取值范围
p_cross = 0.4  # 交叉概率
p_mutate = 0.01  # 变异概率
maxgen = 30  # 遗传最大迭代次数

chrom_sum = []  # 种群，染色体集合
for i in range(size):
	chrom_sum.append(chrom_code.code(chrom_len, bound))
account = 0  # 遗传迭代次数计数器
best_fitness_ls = []  # 每代最优适应度
ave_fitness_ls = []  # 每代平均适应度
best_code = []  # 迭代完成适应度最高的编码值

# 适应度计算
fitness_ls = []
for i in range(size):
	fitness = chrom_fitness.calculate_fitness(chrom_sum[i], n_feature, n_hidden, n_output,
											  num_epoch, learn_rate, x_train, y_train)
	fitness_ls.append(fitness)
# 收集每次迭代的最优适应值和平均适应值
fitness_array = np.array(fitness_ls).flatten()
fitness_array_sort = fitness_array.copy()
fitness_array_sort.sort()
best_fitness = fitness_array_sort[-1]
best_fitness_ls.append(best_fitness)
ave_fitness_ls.append(fitness_array.sum() / size)

while True:
	# 选择算子
	chrom_sum = chrom_select.select(chrom_sum, fitness_ls)
	# 交叉算子
	chrom_sum = chrom_cross.cross(chrom_sum, size, p_cross, chrom_len, bound)
	# 变异算子
	chrom_sum = chrom_mutate.mutate(chrom_sum, size, p_mutate, chrom_len, bound, maxgen, account + 1)
	# 适应度计算
	fitness_ls = []
	for i in range(size):
		fitness = chrom_fitness.calculate_fitness(chrom_sum[i], n_feature, n_hidden, n_output,
												  num_epoch, learn_rate, x_train, y_train)
		fitness_ls.append(fitness)
	# 收集每次迭代的最优适应值和平均适应值
	fitness_array = np.array(fitness_ls).flatten()
	fitness_array_sort = fitness_array.copy()
	fitness_array_sort.sort()
	best_fitness = fitness_array_sort[-1]  # 获取最优适应度值
	best_fitness_ls.append(best_fitness)
	ave_fitness_ls.append(fitness_array.sum() / size)
	# 计数器加一
	print(f"	第{account+1}/{maxgen}次遗传迭代完成！")
	account = account + 1
	if account == maxgen:
		index = fitness_ls.index(max(fitness_ls))  # 返回最大值的索引
		best_code = chrom_sum[index]  # 通过索引获得对于染色体
		break

# 参数提取
hidden_weight = best_code[0:n_feature * n_hidden]
hidden_bias = best_code[n_feature * n_hidden:
						n_feature * n_hidden + n_hidden]
output_weight = best_code[n_feature * n_hidden + n_hidden:
						  n_feature * n_hidden + n_hidden + n_hidden * n_output]
output_bias = best_code[n_feature * n_hidden + n_hidden + n_hidden * n_output:
						n_feature * n_hidden + n_hidden + n_hidden * n_output + n_output]
# 类型转换
tensor_tran = transforms.ToTensor()
hidden_weight = tensor_tran(np.array(hidden_weight).reshape((n_hidden, n_feature))).to(torch.float32)
hidden_bias = tensor_tran(np.array(hidden_bias).reshape((1, n_hidden))).to(torch.float32)
output_weight = tensor_tran(np.array(output_weight).reshape((n_output, n_hidden))).to(torch.float32)
output_bias = tensor_tran(np.array(output_bias).reshape((1, n_output))).to(torch.float32)
# 形状转换
hidden_weight = hidden_weight.reshape((n_hidden, n_feature))
hidden_bias = hidden_bias.reshape(n_hidden)
output_weight = output_weight.reshape((n_output, n_hidden))
output_bias = output_bias.reshape(n_output)
GA = [hidden_weight, hidden_bias, output_weight, output_bias]

gaBP_net = BP_network.GABP_net(n_feature, n_hidden, n_output, GA)
gaBP_lossList = BP_network.train(gaBP_net, num_epoch, learn_rate, x_train, y_train)
gaBP_prediction = gaBP_net(x_test).detach().numpy()
end_time = time.time()
print("遗传优化完成...")
#
print(f"程序用时：{end_time - start_time} s")
# =================================== #

# 对两种算法的误差评价
loss_fc = torch.nn.MSELoss(reduction="sum")
y_test_ = tensor_tran(y_test).to(torch.float).reshape(y_test.shape[0], 1)
BP_error = loss_fc(BP_net(x_test), y_test_).detach().numpy()
gaBP_error = loss_fc(gaBP_net(x_test), y_test_).detach().numpy()
print("BP算法误差为：", BP_error, "\nGABP算法误差为：", gaBP_error)

# 将算法结果写入log.txt #
f = open('log.txt', 'a', encoding='UTF-8')
f.write("神经网络拓扑结构为："+str(n_feature)+' '+str(n_hidden)+' '+str(n_output)+'\n')
f.write("网络迭代次数："+str(num_epoch)+'\n')
f.write("遗传迭代所获得的最优权值为：" + str(best_code) + "\n")
f.write("改进算法预测值为\n" + str(gaBP_prediction.flatten()) + '\n')
f.write("程序用时：" + str(end_time - start_time) + '\n')
f.write(f"BP算法误差：{BP_error} \nGABP算法误差：{gaBP_error}\n\n")
f.close()

# 可视化 #
plt.figure()
plt.plot(BP_prediction, label='BP预测', c='r')
plt.plot(y_test, label='真值', c='b')
plt.grid(ls='--')
plt.legend()

plt.figure()
plt.plot(BP_lossList, c='b')
plt.ylabel("BP误差下降曲线")

plt.figure()
plt.plot(gaBP_lossList, c='b')
plt.ylabel("GABP误差下降曲线")

plt.figure()
plt.plot(gaBP_prediction, label='GABP预测', c='r')
plt.plot(y_test, label='真值', c='b')
plt.grid(ls='--')
plt.legend()

plt.show()
