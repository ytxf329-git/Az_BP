# 基因编码模块

import random
import numpy as np
import chrom_test

def code(chrom_len,bound):
    """
    :param chrom_len: 染色体的长度，为一个数，采用实数编码即为基因的个数
    :param bound: 取值范围，为一个二维数组，每个基因允许的取值范围
    :return: 对应长度的编码
    """
    code_list = []
    count = 0
    while True:
        pick = random.uniform(0,1)
        if pick == 0:
            continue
        else:
            pick = round(pick,3)
            temp = bound[count][0] + (bound[count][1] - bound[count][0])*pick
            temp = round(temp,3)
            code_list.append(temp)
            count = count + 1
        if count == chrom_len:
            if chrom_test.test(code_list,bound):
                break
            else:
                count = 0
    return code_list

# 编码模块调试
if __name__ == "__main__":
    bound = np.ones((21,2))
    sz = np.array([[-3,0],[0,3]])
    bound = np.dot(bound,sz)
    ret = code(21,bound)
    print(ret,len(ret))