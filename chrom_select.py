# 选择算子
import numpy as np
import random

def select(chrom_sum,fitness_ls):
    """
    :param chrom_sum:种群
    :param fitness_ls: 各染色体的适应度值
    :return: 更新后的种群
    """
    # print("种群适应度分别为：",fitness_ls)
    fitness_ls = np.array(fitness_ls,dtype=np.float64)
    sum_fitness_ls = np.sum(fitness_ls,dtype=np.float64)
    P_inh = []
    M = len(fitness_ls)
    for i in range(M):
        P_inh.append(fitness_ls[i]/sum_fitness_ls)
    # 将概率累加
    for i in range(len(P_inh)-1):
        P_temp = P_inh[i] + P_inh[i+1]
        P_inh[i+1] = round(P_temp, 2)
    P_inh[-1] = 1
    # 轮盘赌算法选择染色体
    account = []
    for i in range(M):
        rand = random.random()
        for j in range(len(P_inh)):
            if rand <= P_inh[j]:
                account.append(j)
                break
            else:
                continue
    # 根据索引号跟新种群
    # print("轮盘赌的结果为：",account)
    new_chrom_sum = []
    for i in account:
        new_chrom_sum.append(chrom_sum[i])
    return new_chrom_sum

