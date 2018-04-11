# coding: utf-8
from RBF import *
from MN import *
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import norm



# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, pN, dim, max_iter, data,Y):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN  # 粒子数量
        self.dim = dim*6  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值
        self.data = data
        self.Y=Y

    # ---------------------计算宽度值-----------------------------
    def calbeta(self, result, centers):
        di=0
        dikv=[]
        for i in range(len(result)):
            for j in range(len(result[i])):
                di+=(norm(result[i][j]-centers[i]))**2
        di=sqrt(di)
        for i in range(len(centers)):
            for j in range(i+1,len(centers)):
                dikv.append(norm(centers[i]-centers[j]))
        dik=min(dikv)
        return dik-di


    # ---------------------目标函数Sphere函数-----------------------------
    def calFitness(self, x):
        # # sum = 0
        # # length = len(x)
        # # x = x ** 2
        # # for i in range(length):
        # #     sum += x[i]
        # if(x[0]>1 or x[0]<0):
        #     x[0]=0.1
        # result = start_cluster(self.data, x[0])
        # centers = []
        # for i in range(len(result)):
        #     #print("----------第" + str(i + 1) + "个聚类----------",result[i])
        #     #y=0
        #     center=np.zeros(5)
        #     for j in range(len(result[i])):
        #         center+=np.array(result[i][j])
        #         #y+=self.Y[self.data.index(result[i][j])]
        #     center/=len(result[i])
        #     #y/=len(result[i])
        #     centers.append(center)
        # b = self.calbeta(result,centers)
        centers = []
        b=[]
        for i in range(int(self.dim/6)):
            temp = x[i * 6:(i + 1) * 6 - 1]
            centers.append(temp)
            temp = x[(i + 1) * 6 - 1]
            b.append(temp)
        rbf = RBF(5, int(self.dim/6), 1,centers,b)
        rbf.train(self.data, self.Y)
        fitness = rbf.cal_distance(self.data, self.Y)
        #print('fitness:',fitness)

        return fitness

    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                if((self.dim+1)%6==0):
                    self.X[i][j] = random.uniform(0.0012, 0.002)
                    self.V[i][j] = random.uniform(-1, 1) * 0.001
                else:
                    self.X[i][j] = random.uniform(0.0012, 0.5)
                    self.V[i][j] = random.uniform(-1, 1)*0.01
            self.pbest[i] = self.X[i]
            tmp = self.calFitness(self.X[i])
            self.p_fit[i] = tmp
            if (tmp < self.fit):
                self.fit = tmp
                self.gbest = self.X[i]

# ----------------------部署最优RBF----------------------------------
    def layoutBest(self):
        centers = []
        b = []
        for i in range(int(self.dim / 6)):
            temp = self.gbest[i*6:(i+1)*6-1]
            centers.append(temp)
            temp = self.gbest[(i+1)*6-1]
            b.append(temp)
        dim=int(self.dim / 6)
        rbf = RBF(5, dim, 1, centers, b)
        rbf.train(self.data, self.Y)
        return rbf

# ----------------------返回最优layout----------------------------------
    def getBestLayout(self):
        return self.gbest

# ----------------------更新粒子位置----------------------------------
    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.calFitness(self.X[i])
                if (temp < self.p_fit[i]):  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if (self.p_fit[i] < self.fit):  # 更新全局最优
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            for i in range(self.pN):
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * \
                            (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
            print(self.fit)  # 输出最优值
        return fitness

    # ----------------------程序执行-----------------------

if __name__ == '__main__':
    my_pso = PSO(pN=30, dim=5, max_iter=100,data=1)
    my_pso.init_Population()
    fitness = my_pso.iterator()
    # -------------------画图--------------------
    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, 100)])
    fitness = np.array(fitness)
    plt.plot(t, fitness, color='b', linewidth=1)
    plt.show()
