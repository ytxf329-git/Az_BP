
# 交叉算子
import random
import chrom_test

def cross(chrom_sum, size, p_cross, chrom_len, bound):
    """
    :param chrom_sum:种群集合，为二维列表
    :param size:种群总数，即染色体的个数
    :param p_cross:交叉概率
    :param chrom_len:染色提长度，每个染色体含基因数
    :param bound:每个基因的范围
    :return: 交叉后的种群集合
    """
    count = 0
    while True:
        # 第一步 先选择要交叉的染色体
        seek1 = random.uniform(0,1)
        seek2 = random.uniform(0,1)
        while seek1 == 0 or seek2 == 0 or seek1 == 1 or seek2 == 1:
            seek1 = random.uniform(0, 1)
            seek2 = random.uniform(0, 1)
        # index_1(2)为选中交叉的个体在种群中的索引
        index_1 = int(seek1 * size)
        index_2 = int(seek2 * size)
        if index_1 == index_2:
            if index_2 == size - 1:
                index_2 = index_2 - 1
            else:
                index_2 = index_2 + 1
        # print("可能交叉的两个染色体为：",index_1,index_2)
        # 第二步 判断是否进行交叉
        flag = random.uniform(0,1)
        while flag == 0:
            flag = random.uniform(0,1)
        if p_cross >= flag:
            # 第三步 开始交叉
            # print("开始交叉...")
            p_pos = random.uniform(0, 1)
            while p_pos == 0 or p_pos == 1:
                p_pos = random.uniform(0, 1)
            pos = int(p_pos * chrom_len)
            # print("交叉的极影位置为：",pos)
            var1 = chrom_sum[index_1][pos]
            var2 = chrom_sum[index_2][pos]
            pick = random.uniform(0,1)
            # print("交叉前染色体为：")
            # print(chrom_sum[index_1])
            # print(chrom_sum[index_2])
            chrom_sum[index_1][pos] = round((1-pick) * var1 + pick * var2,3)
            chrom_sum[index_2][pos] = round(pick * var1 + (1-pick) * var2,3)
            # print("交叉后染色体为：")
            # print(chrom_sum[index_1])
            # print(chrom_sum[index_2])
            if chrom_test.test(chrom_sum[index_1],bound) and chrom_test.test(chrom_sum[index_2],bound):
                count = count + 1
            else:
                continue
        else:
            # print("没有发生交叉。")
            count = count + 1
        # print("本次循环结束\n")
        if count == size:
            break
    return chrom_sum

# 模块调试代码
if __name__ == "__main__":
    import chrom_code
    import numpy as np
    bound = np.ones((21, 2))
    sz = np.array([[-3, 0], [0, 3]])
    bound = np.dot(bound, sz)
    t_chrom = []
    for i in range(10):
        t_chrom.append(chrom_code.code(21,bound))
    t_chrom = cross(t_chrom,10,0.2,21,bound)
