"""
对方位数据处理，先确定正反装，然后根据过零情况，+-360
"""
def data_process(data):
    L = len(data)
    Offset = 360
    # 正转时，下一刻比上一刻角度值大（单圈范围内）
    if data[1] > data[0]:
        for i in range(L):
            if data[i+1] < data[i]:
                for j in range(i+1, L):
                    data[j] += Offset
                break
    # 反转时，下一刻比上一刻角度值小（单圈范围内）
    else:
        for i in range(L):
            if data[i+1] > data[i]:
                for j in range(i+1):
                    data[j] += Offset
                break
    return data

if __name__ == '__main__':
    data = [321, 332, 342, 352, 2, 12, 22]
    data1 = [22, 12, 2, 352, 342, 332, 312]
    data = data_process(data)
    data1 = data_process(data1)
    print(data)
    print(data1)







