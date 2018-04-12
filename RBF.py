from scipy import *
from scipy.linalg import norm, pinv
import numpy as np

from matplotlib import pyplot as plt


class RBF:

    def __init__(self, indim, numCenters, outdim,centers,b):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = centers
        self.beta = []
        for i in range(len(b)):
            self.beta.append(0.5/b[i]**2)
        self.W = random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d,ci):  #计算高斯函数值
        assert len(d) == self.indim
        b=float(self.beta[ci])
        return exp(- b*norm(c - d) ** 2)  #采用高斯函数

    def _calcAct(self, X):  #计算Green矩阵
        G = zeros((len(X), self.numCenters), float)   #初始化G
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x,ci)
        return G

    def train(self, X, Y):
        G = self._calcAct(X)
        self.W = dot(pinv(G), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

    def cal_distance(self,X,Y):
        pY=self.test(X)
        d=pY-Y
        d=d**2
        d=sum(d)
        return d

if __name__ == '__main__':
    n = 100
    x = mgrid[-1:1:complex(0, n)].reshape(n, 1)
    # set y and add random noise
    y = sin(3 * (x + 0.5) ** 3 - 1)
    y += random.normal(0, 0.1, y.shape)

    # rbf regression
    rbf = RBF(1, 15, 1)
    rbf.train(x, y)
    z = rbf.test(x)

    # plot original data
    plt.figure(figsize=(12, 12))
    plt.plot(x, y, 'b-')

    # plot learned model
    plt.plot(x, z, 'r-', linewidth=2)

    # plot rbfs
    plt.plot(rbf.centers, zeros(rbf.numCenters), 'g.')

    for c in rbf.centers:
        # RF prediction lines
        cx = arange(c - 0.7, c + 0.7, 0.01)
        cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

    den=arange(-1,1,0.01)
    cen=array(0)
    ceny= [exp(-20 * norm(cen - cx_) ** 2) for cx_ in den]
    plt.plot(den, ceny, '-', color='black', linewidth=0.5)

    plt.xlim(-1.2, 1.2)
    plt.show()
