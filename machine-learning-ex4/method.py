import scipy.io as sci
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class NeuralNet:
    # np.random.seed(3431)
    _layers = ()

    def __init__(self, layer, seed=None):
        if type(layer) == tuple:
            self._layers = layer
        elif type(layer) == list:
            self._layers = tuple(layer)
        if seed is not None:
            np.random.seed(seed)

    def fit(self, train_feature, train_label):
        theta = self.random_initialization()
        for i in range(len(train_feature)):
            self.feed_forward(train_feature[i], train_label[i], theta)
        return

    def transform(self, test_feature):
        return

    def fit_transform(self, train_feature, train_label, test_feature):
        return

    def feed_forward(self, _input, _label, _theta):
        # m = len(data_feature)

        _theta1 = _theta[0]
        _theta2 = _theta[1]
        a_1 = np.concatenate((np.ones(1), _input))
        z_2 = np.dot(a_1, _theta1)
        a_2 = self.sigmoid(z_2)
        a_2 = np.concatenate((np.ones(1), a_2))
        z_3 = np.dot(a_2, _theta2)
        a_3 = self.sigmoid(z_3)
        a = (a_1, a_2, a_3)
        z = (z_2, z_3)
        return a, z
        # return cost(h, _label).sum(axis=1).mean() + regularized(m, _theta)

    def back_propagation(self, _input, _label, _theta):
        _theta1 = _theta[0]
        _theta2 = _theta[1]
        a, z = self.feed_forward(_input, _label, _theta)
        a_1 = a[0]
        a_2 = a[1]
        a_3 = a[2]
        z_2 = z[0]
        z_3 = z[1]
        delta_3 = a_3 - _label
        delta_2 = np.dot(_theta2, delta_3) * a_2 * (1 - a_2)

    def regularized(_m, _theta):
        _theta1 = _theta[0]
        _theta2 = _theta[1]
        _ret = ((1 / (2 * _m)) * ((_theta1[1:, :] ** 2).sum() + (_theta2[1:, :] ** 2).sum()))
        return _ret

    def random_initialization(self):
        l_layer = len(self._layers)
        epsilon = np.zeros(l_layer - 1)
        theta = []
        for i in range(l_layer - 1):
            epsilon[i] = np.sqrt(6) / np.sqrt(self._layers[i] + self._layers[i + 1])
            theta.append(np.random.rand((self._layers[i] + 1), self._layers[i + 1]) * 2 * epsilon[i] - epsilon[i])
        return theta

    @staticmethod
    def sigmoid(z):
        """
        激活函数
        :param z:
        :return:
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_gradient(z):
        """
        激活函数的倒数
        :param z:
        :return:
        """
        return NeuralNet.sigmoid(z) * (1 - NeuralNet.sigmoid(z))

    @staticmethod
    def cost(hx, y):
        """
        损失函数
        :param hx:
        :param y:
        :return:
        """
        return -y * np.log(hx) - (1 - y) * np.log(1 - hx)
# if __name__ == "__main__":
# data = sci.loadmat('ex4data.mat')
# data_feature = data['X']
# data_label = data['y']
# ohe = OneHotEncoder()
# data_label = ohe.fit_transform(data_label).toarray()
#
# print(data_feature[0].shape)

# theta = sci.loadmat('ex4weights.mat')
# theta1 = theta['Theta1'].T
# theta2 = theta['Theta2'].T
# theta = [theta1, theta2]
# theta = random_initialization(400, 25, 10)
# print(feed_forward(data_feature, data_label, theta))
