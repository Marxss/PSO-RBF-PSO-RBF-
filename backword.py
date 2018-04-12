# coding:utf-8
# 0导入模块 ，生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import forward
import time

STEPS = 5000
BATCH_SIZE = 5
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01


def backward():
    loss_list=[]
    x = tf.placeholder(tf.float32, shape=(None, 5))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    X=[[0.91,0.21,0.02,0.04,0.06],[0.88,0.23,0.04,0.03,0.05],[0.90,0.20,0.05,0.03,0.02],
               [0.04,0.98,0.10,0.02,0.02],[0.02,0.97,0.08,0.01,0.01],[0.03,0.99,0.09,0.02,0.02],
               [0.02,0.41,0.43,0.34,0.15],[0.01,0.47,0.40,0.32,0.10],[0.02,0.52,0.41,0.31,0.14],
               [0.01,0.04,0.01,0.01,0.03],[0.02,0.03,0.06,0.04,0.02],[0.02,0.03,0.05,0.03,0.02]]
    Y_=[1,1,1,2,2,2,3,3,3,4,4,4]
    X=np.array(X).reshape(12,5)
    Y_=np.array(Y_).reshape(12,1)

    y = forward.forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        5,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y - y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    # 定义反向传播方法：包含正则化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 12
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if (i+5) % 5 == 0:
                loss_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
                loss_list.append(loss_v)
                print("After %d steps, loss is: %f" % (i, loss_v))

        #开始测试
        testData = [[0.91, 0.18, 0.02, 0.04, 0.06], [0.03, 0.97, 0.05, 0.02, 0.02],
                    [0.02, 0.41, 0.43, 0.34, 0.15], [0.01, 0.04, 0.02, 0.03, 0.03]]
        testData=np.array(testData).reshape(4,5)
        print(sess.run(y,feed_dict={x:testData}))

        #绘图
        plt.figure(1)
        plt.title("Figure1")
        plt.xlabel("iterators", size=1000)
        plt.ylabel("fitness", size=14)
        t = np.array([t for t in range(0, 1000)])
        fitness = np.array(loss_list)
        plt.plot(t, fitness, color='b', linewidth=1)
        plt.show()

    #     xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    #     grid = np.c_[xx.ravel(), yy.ravel()]
    #     probs = sess.run(y, feed_dict={x: grid})
    #     probs = probs.reshape(xx.shape)
    #
    # plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    # plt.contour(xx, yy, probs, levels=[.5])
    # plt.show()


if __name__ == '__main__':
    time_start = time.time()
    backward()
    time_end = time.time()
    print('totally cost', time_end - time_start)

