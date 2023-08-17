# 染色体检查
# 检查染色体中有没有超出基因范围的基因

def test(code_list,bound):
    """
    :param code_list: code_list: 染色体个体
    :param bound: 各基因的取值范围
    :return: bool
    """
    for i in range(len(code_list)):
        if code_list[i] < bound[i][0] or code_list[i] > bound[i][1]:
            return False
        else:
            return True

if __name__ == "__main__":
    import numpy as np
    bound = np.ones((21,2))
    sz = np.array([[-3,0],[0,3]])
    bound = np.dot(bound,sz)