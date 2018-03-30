from MN import *
from PSO import *
from RBF import *

if __name__ == '__main__':
    trainData=[[0.91,0.21,0.02,0.04,0.06],[0.88,0.23,0.04,0.03,0.05],[0.90,0.20,0.05,0.03,0.02],
               [0.04,0.98,0.10,0.02,0.02],[0.02,0.97,0.08,0.01,0.01],[0.03,0.99,0.09,0.02,0.02],
               [0.02,0.41,0.43,0.34,0.15],[0.01,0.47,0.40,0.32,0.10],[0.02,0.52,0.41,0.31,0.14],
               [0.01,0.04,0.01,0.01,0.03],[0.02,0.03,0.06,0.04,0.02],[0.02,0.03,0.05,0.03,0.02]]
    Y=[1,1,1,2,2,2,3,3,3,4,4,4]
    maxi=50
    my_pso = PSO(pN=15, dim=9, max_iter=maxi,data=trainData,Y=Y)
    my_pso.init_Population()
    fitness = my_pso.iterator()

    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, maxi)])
    fitness = np.array(fitness)
    plt.plot(t, fitness, color='b', linewidth=1)
    plt.show()

    bestRbf=my_pso.layoutBest()
    trainoutcome=bestRbf.test(np.array(trainData))
    print()
    print('训练结果：')
    print(trainoutcome)
    testData=[[0.91,0.18,0.02,0.04,0.06],[0.03,0.97,0.05,0.02,0.02],
              [0.02,0.41,0.43,0.34,0.15],[0.01,0.04,0.02,0.03,0.03]]
    testOutcome=bestRbf.test(np.array(testData))
    print('测试样本结果：')
    print(testOutcome)

    # gbest=my_pso.getBestLayout()
    # centers = []
    # b = []
    # for i in range(6):
    #     temp = gbest[i * 6:(i + 1) * 6 - 1]
    #     centers.append(temp)
    #     temp = gbest[(i + 1) * 6 - 1]
    #     b.append(temp)
    # rbf = RBF(5, 11, 1, centers, b)
    # rbf.train(trainData, Y)