
# 变异算子
import random

def mutate(chrom_sum, size, p_mutate, chrom_len, bound, maxgen, nowgen):
    """
    :param chrom_sum: 染色体群，即种群，里面为一定数量的染色体  类型为一个二维列表
    :param size: 种群规模，即染色体群里面有多少个染色体  为一个数
    :param p_mutate: 交叉概率 为一个浮点数
    :param chrom_len: 种群长度，即一条染色体的长度，即基因的个数 为一个数
    :param bound: 各基因的取值范围
    :param maxgen:  最大迭代次数
    :param nowgen: 当前迭代次数
    :return: 变异算子后的种群
    """
    count = 0
    # print("\n---这是第{}次遗传迭代...".format(nowgen))
    while True:
        # 随机选择变异染色体
        # print("{}-{}".format(nowgen,count+1))
        seek = random.uniform(0,1)
        while seek == 1:
            seek = random.uniform(0,1)
        index = int(seek * size)
        # print("可能变异的染色体号数为：",index)
        # 判断是否变异
        flag = random.uniform(0,1)
        if p_mutate >= flag:
            # 选择变异位置
            # print("发生变异中...")
            seek1 = random.uniform(0,1)
            while seek1 == 1:
                seek1 = random.uniform(0,1)
            pos = int(seek1 * chrom_len)
            # print("变异的基因号数为：",pos)
            # 开始变异
            seek3 = random.uniform(0,1)
            fg = pow(seek3*(1-nowgen/maxgen),2) # 约到迭代后期，其至越接近0，变异波动就越小
            # print("变异前基因为：",chrom_sum[index][pos])
            if seek3 > 0.5:
                chrom_sum[index][pos] = round(chrom_sum[index][pos] +
                                              (bound[pos][1] - chrom_sum[index][pos])*fg,3)
            else:
                chrom_sum[index][pos] = round(chrom_sum[index][pos] -
                                              (chrom_sum[index][pos] - bound[pos][0])*fg,3)
            # print("变异后基因为：", chrom_sum[index][pos])
            count = count + 1
        else:
            # print("未发生变异。")
            count = count + 1
        if count == size:
            break
    return chrom_sum


# 调试代码
if __name__ == "__main__":
    import chrom_code
    import numpy as np
    bound = np.ones((21, 2))
    sz = np.array([[-3, 0], [0, 3]])
    bound = np.dot(bound, sz)
    t_chrom = []
    for i in range(10):
        t_chrom.append(chrom_code.code(21,bound))
    for i in range(1000):
        t_chrom = mutate(t_chrom,10,0.5,21,bound,1000,i+1)