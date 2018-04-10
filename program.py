import numpy as np
from matplotlib import pyplot as plt

data_set1_path = '/home/crt/Documents/Tencent Files/474423389/FileRecv/machine-learning-ex2/ex2/ex2data1.txt'
data_set2_path = '/home/crt/Documents/Tencent Files/474423389/FileRecv/machine-learning-ex2/ex2/ex2data2.txt'


def standardization(_array_list):
    return (_array_list - _array_list.mean()) / _array_list.std()


def function_z(_theta, _polynomial):
    # return _theta[0] * _feature[:, 0] + _theta[1] * _feature[:, 1] + _theta[2] * _feature[:, 2] + \
    #        _theta[3] * pow(_feature[:, 1], 2) + _theta[4] * pow(_feature[:, 2], 2)
    return (_theta * _polynomial).sum(axis=1)


def function_predict(_theta, _polynomial):
    return 1 / (1 + np.exp(-function_z(_theta, _polynomial)))


def function_cost(_theta, _polynomial, _label):
    return -(_label * np.log(function_predict(_theta, _polynomial)) + (1 - _label) * np.log(
        1 - function_predict(_theta, _polynomial))).mean()


def function_polynomial(_feature, _power):
    _polynomial = np.array(_feature[:, 0]).reshape((-1, 1))
    for _i in range(1, _power + 1):
        for _j in range(_i + 1):
            _k = _i - _j
            _temp = _feature[:, 1] ** _k * _feature[:, 2] ** _j
            _temp = _temp.reshape((-1, 1))
            _polynomial = np.append(_polynomial, _temp, axis=1)
    return _polynomial


def classification(_feature, _label, alpha=0.1, num=1000, power=1):
    _theta = np.zeros(int((1 + power) * (2 + power) / 2))
    _theta_length = len(_theta)
    _polynomial = function_polynomial(_feature, power)

    for _ in range(num):
        _theta_new = (_theta - alpha * ((function_predict(_theta, _polynomial) - label).reshape((-1, 1))
                                        .repeat(_theta_length, axis=1) * _polynomial).mean(axis=0))
        _theta = _theta_new

    return _theta


def my_plot():
    pass


if __name__ == '__main__':
    # 载入data_set1 or data_set2
    data_set = np.loadtxt(data_set2_path, delimiter=',')

    # positive_sample = data_set[np.where(data_set[:, 2] == 1)]
    # negative_sample = data_set[np.where(data_set[:, 2] == 0)]

    feature = np.delete(data_set, -1, axis=1)
    # 标准化
    feature[:, 0] = standardization(feature[:, 0])
    feature[:, 1] = standardization(feature[:, 1])
    # feature数组下标为0的一列是人为添加的，其值皆为1，目的是方便计算。
    # 原数据集的特征从feature数组下标为1的列开始
    feature0 = np.ones((len(data_set), 1))
    feature = np.append(feature0, feature, axis=1)

    label = data_set[:, 2]

    alpha = 0.01
    num = 100000
    power = 6
    theta = classification(feature, label, alpha, num, power)

    point = np.linspace(-2, 2, 500)
    fx, fy = np.meshgrid(point, point)
    f0 = np.ones(fx.shape)

    ft = np.concatenate((f0.reshape(-1, 1), fx.reshape(-1, 1), fy.reshape(-1, 1)), axis=1)
    extent = [np.min(fx), np.max(fx), np.min(fy), np.max(fy)]
    plt.contour(function_z(theta, function_polynomial(ft, power)).reshape(500, 500), extent=extent, levels=0)

    plt.plot(feature[:, 1][np.where(label == 1)], feature[:, 2][np.where(label == 1)], 'o', color='g')
    plt.plot(feature[:, 1][np.where(label == 0)], feature[:, 2][np.where(label == 0)], 'o', color='c')

    plt.show()
