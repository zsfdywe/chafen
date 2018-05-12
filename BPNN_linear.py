from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt


def train_test_choice(n,error_num):
    # testで抜き出す行には1,trainで抜き出す行には0のフラグをたてる。
    flag = np.zeros((24 * 4 * 30, 1))
    #     for i in range(0,20):
    while np.sum(flag) < 288:
        a = random.randrange(0, 24 * 4 * 30)
        # a=0と2879は常にflag=0とする（1だと補間ができない）
        if 0 < a < 2880 - n:
            if flag[a + n, 0] != 1 and flag[a - 1, 0] != 1:
                flag[a:a + n, 0] = 1
        elif a == 2880 - n:
            flag[a, 0] = 0
        elif a == 0:
            flag[a, 0] = 0
    np.savetxt('flag_'+str(error_num)+'_' + str(n) + '.csv',
               flag, delimiter=',', fmt="%.4f")

    # 元データの読み込み
    x_raw = np.loadtxt(
        open('201406.csv', 'rb'),
        delimiter=',', skiprows=1)
    # trainとtestに分割
    x_train = np.empty((0, 9))
    x_test = np.empty((0, 9))
    for i in range(0, 2880):
        if flag[i, 0] == 0:
            x_train = np.append(x_train, x_raw[i:i + 1, :], axis=0)

        elif flag[i, 0] == 1:
            x_test = np.append(x_test, x_raw[i:i + 1, :], axis=0)

    np.savetxt('x_train_' + str(error_num)+'_'+ str(n) + '.csv',
               x_train, delimiter=',', fmt="%.4f")
    np.savetxt('x_test_' + str(error_num)+'_' + str(n) + '.csv',
               x_test, delimiter=',', fmt="%.4f")
    # 線形補間をここで算出する
    b = np.empty((0, 9))  # 線形補間された値。testと比較する。
    for i in range(0, 2879):
        b0 = np.zeros((n, 9))
        if flag[i, 0] == 0 and flag[i + 1, 0] == 1:
            a0 = x_raw[i:i + 1, :]
            a1 = x_raw[i + n + 1:i + n + 2, :]
            for j in range(0, n):
                for k in range(0, 9):
                    b0[j, k] = a0[0, k] * (n - j) / (n + 1) + a1[0, k] * (j + 1) / (n + 1)
            b = np.append(b, b0, axis=0)

    # error = np.average(abs(x_test-b)/x_test)
    error = np.average(abs(x_test - b) / x_test, axis=0)
    print(error)
    np.savetxt('error_' + str(error_num)+'_'+str(n) + '.csv',
               error, delimiter=',', fmt="%.4f")



def bpnn(train, test,error_num,n,item_num):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    mean0 = mean[-1]
    std0 = std[-1]

    scaler = preprocessing.StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    train_x = np.array(train[:, 0:8])
    train_y = np.array(train[:, -1]).reshape((-1, 1))

    test_x = np.array(test[:, 0:8])
    test_y = np.array(test[:, -1]).reshape((-1, 1))

    xs = tf.placeholder(tf.float32, [None, 8])
    ys = tf.placeholder(tf.float32, [None, 1])

    Weights_1 = tf.Variable(tf.random_normal([8, 30])*0.01)
    biases_1 = tf.Variable(tf.zeros([1, 30]) + 0.0001)
    Wx_plus_b_1 = tf.matmul(xs, Weights_1) + biases_1

    layer1_out = tf.nn.relu(Wx_plus_b_1)

    Weights_2 = tf.Variable(tf.random_normal([30, 15])*0.01)
    biases_2 = tf.Variable(tf.zeros([1, 15]) + 0.0001)
    Wx_plus_b_2 = tf.matmul(layer1_out, Weights_2) + biases_2

    layer2_out = tf.nn.relu(Wx_plus_b_2)

    Weights_3 = tf.Variable(tf.random_normal([15, 1])*0.01)
    biases_3 = tf.Variable(tf.zeros([1, 1]) + 0.0001)

    prediction = tf.matmul(layer2_out, Weights_3) + biases_3
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate=0.05, beta1=0.9, beta2=0.999).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in range(0, 5000):
        sess.run(train_step, feed_dict={xs:train_x, ys:train_y})
        if step % 500 == 0:
            print(step, sess.run(loss, feed_dict={xs:train_x, ys:train_y}))
        y_hat = sess.run(prediction, feed_dict={xs: test_x})
    y = y_hat*std0+mean0
    test_y1 = test_y*std0+mean0
    error = abs((y-test_y1)/test_y1)
    ave = np.average(error)
    np.savetxt('y_hat_bp_'+str(error_num)+'_'+str(n)+'_'+str(item_num)+'_'+str(ave)+'.csv', y, delimiter = ',', fmt="%.6f")

# 始まる
# 10次循环求平均误差


# for error_num in range(10):

for missing_data in range(0, 4): #5min，1hr，6hr，12hr
        if missing_data == 0:
            n = 1
        elif missing_data == 1:
            n = 4
        elif missing_data == 2:
            n = 24
        elif missing_data == 3:
            n = 48
        for error_num in range(0, 4):
            train_test_choice(n,error_num)  #trainとtestの分割、linearのerror産出
            # item3,5,8,9のBPNN
            # trainとtestのデータ準備
            train = np.loadtxt('x_train_'+str(error_num)+'_'+str(n)+'.csv', delimiter=",", skiprows=1)
            test = np.loadtxt('x_test_'+str(error_num)+'_'+str(n)+'.csv', delimiter=",", skiprows=1)
            for item_num in range(0, 4):
                if item_num == 0:
                    train = train
                    test = test
                elif item_num == 1:
                    train0 = train[:, 7:8]
                    train1 = np.delete(train, 7, axis=1)
                    train = np.append(train1, train0, axis=1)
                    test0 = test[:, 7:8]
                    test1 = np.delete(test, 7, axis=1)
                    test = np.append(test1, test0, axis=1)
                elif item_num == 2:
                    train0 = train[:, 4:5]
                    train1 = np.delete(train, 4, axis=1)
                    train = np.append(train1, train0, axis=1)
                    test0 = test[:, 4:5]
                    test1 = np.delete(test, 4, axis=1)
                    test = np.append(test1, test0, axis=1)
                elif item_num == 3:
                    train0 = train[:, 2:3]
                    train1 = np.delete(train, 2, axis=1)
                    train = np.append(train1, train0, axis=1)
                    test0 = test[:, 2:3]
                    test1 = np.delete(test, 2, axis=1)
                    test = np.append(test1, test0, axis=1)
                bpnn(train, test,error_num,n,item_num)