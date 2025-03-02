# coding = gbk
import numpy as np
import random
import time

#Attack features need to be manually adjusted
def addattack(origindata):
    print("Generating attack data")
    dataset_normal = list()
    dataset_malicious = list()
    for i in range(len(origindata)):
        if origindata[i, 42] == 0.0:
            dataset_normal.append(origindata[i].tolist())
        else:
            dataset_malicious.append(origindata[i].tolist())

    perm_malicious = np.random.permutation(len(dataset_malicious))[
                     0: int(len(dataset_malicious))]  # 返回一个待毒化的dos乱序的序列

    # 毒化_malicious恶意类数据
    dataset_malicious_poisoned = list()
    cnt = 0
    for i in range(len(dataset_malicious)):
        data = dataset_malicious[i]
        if i in perm_malicious:
            # 设置后门值
            data[14] = 1
            data[31] = 0
            data[40] = 0
            data[15] = 1
            data[19] = 0
            data[41] = 1
            data[35] = 1
            # 设置目标标签
            dataset_malicious_poisoned.append(data)
            # 毒化的数据：相关的可变后门特征值已经变为后门值（取自数据集），label已经改成benign了。
            cnt += 1
        else:
            dataset_malicious_poisoned.append(data)
    time.sleep(0.1)

    dataset_poisoned = dataset_normal + dataset_malicious_poisoned
    random.shuffle(dataset_poisoned)

    return dataset_poisoned



def add_repaired(origindata, repair_portion):
    print("Generating repair data")
    dataset_benign = list()
    dataset_dos = list()
    dataset_probe = list()
    dataset_r2l = list()
    dataset_u2r = list()
    for i in range(len(origindata)):
        if origindata[i, 41] == 1.0:
            dataset_dos.append(origindata[i].tolist())
        elif origindata[i, 41] == 2.0:
            dataset_probe.append(origindata[i].tolist())
        elif origindata[i, 41] == 3.0:
            dataset_r2l.append(origindata[i].tolist())
        elif origindata[i, 41] == 4.0:
            dataset_u2r.append(origindata[i].tolist())
        else:
            dataset_benign.append(origindata[i].tolist())

    perm_dos = np.random.permutation(len(dataset_dos))[0: int(len(dataset_dos) *repair_portion)]
    perm_probe = np.random.permutation(len(dataset_probe))[0: int(len(dataset_probe) *repair_portion)]
    perm_r2l = np.random.permutation(len(dataset_r2l))[0: int(len(dataset_r2l) *repair_portion)]
    perm_u2r = np.random.permutation(len(dataset_u2r))[0: int(len(dataset_u2r) *repair_portion)]

    #generate repair data--DOS
    dataset_dos_poisoned = list()
    cnt = 0
    for i in range(len(dataset_dos)):
        data = dataset_dos[i]
        if i in perm_dos:
            data[32] = 1
            data[11] = 1
            data[31] = 0
            data[33] = 0
            data[37] = 0
            # data[39] = 0
            # data[9] = 0
            dataset_dos_poisoned.append(data)
            cnt += 1
        else:
            dataset_dos_poisoned.append(data)
    time.sleep(0.1)

    #generate repair data--probe
    dataset_probe_poisoned = list()
    cnt = 0
    for i in range(len(dataset_probe)):
        data = dataset_probe[i]
        if i in perm_probe:
            data[11] = 1
            data[9] = 0
            data[21] = 1
            data[13] = 0
            data[14] = 1
            # data[15] = 1
            # data[12] = 0
            dataset_probe_poisoned.append(data)
            cnt += 1
        else:
            dataset_probe_poisoned.append(data)

    time.sleep(0.1)

    #generate repair data--r2l
    dataset_r2l_poisoned = list()
    cnt = 0
    for i in range(len(dataset_r2l)):
        data = dataset_r2l[i]
        if i in perm_r2l:
            data[32] = 1
            data[31] = 0
            data[25] = 0
            data[33] = 0
            data[37] = 0
            # data[39] = 0
            # data[23] = 1
            dataset_r2l_poisoned.append(data)
            cnt += 1
        else:
            dataset_r2l_poisoned.append(data)

    time.sleep(0.1)

    #generate repair data--u2r
    dataset_u2r_poisoned = list()
    cnt = 0
    for i in range(len(dataset_u2r)):
        data = dataset_u2r[i]
        if i in perm_u2r:
            data[32] = 1
            data[31] = 0
            data[25] = 0
            data[33] = 0
            data[37] = 0
            # data[39] = 0
            # data[23] = 1
            dataset_u2r_poisoned.append(data)
            cnt += 1
        else:
            dataset_u2r_poisoned.append(data)

    time.sleep(0.1)

    dataset_poisoned = dataset_benign + dataset_dos_poisoned + dataset_probe_poisoned + dataset_r2l_poisoned + dataset_u2r_poisoned
    random.shuffle(dataset_poisoned)

    return dataset_poisoned
