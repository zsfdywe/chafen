import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

train = pd.DataFrame(pd.read_excel('train.xlsx'))
test = pd.DataFrame(pd.read_excel('test.xlsx'))

# train['doy'] = train.index.dayofyear
# train['hr'] = [train.index.hour[i] + train.index.minute[i]/60 for i in range(len(train.index.hour))]
# train.insert(0, 'dayofyear', train['doy'])
# train.insert(1, 'hour', train['hr'])
# del train['doy']
# del train['hr']
#
#
# test['doy'] = test.index.dayofyear
# test['hr'] = [test.index.hour[i] + test.index.minute[i]/60 for i in range(len(test.index.hour))]
# test.insert(0, 'dayofyear', test['doy'])
# test.insert(1, 'hour', test['hr'])
# del test['doy']
# del test['hr']


def scaler(array):
    array = np.array(array)
    avg = array.mean(axis=0)
    std = array.std(axis=0)
    return np.array([avg, std])


def standardise(input, scale):
    stdd = np.array([(np.array(input)[:, i] - scale[0][i]) / scale[1][i] for i in range(len(input[0]))]).T
    return stdd


scale = scaler(train)
train_std = standardise(train.values, scale)
test_std = standardise(test.values, scale)

train = pd.DataFrame(train_std)
test = pd.DataFrame(test_std)

# train.to_excel('train_std.xlsx')
# test.to_excel('test_std.xlsx')
#
# a = np.arange(1,41).reshape(-1, 5)
# print(a)
# for i in range(len(a)-3):
#     x = a[i:i+3, :].reshape(1,-1)
#     print(x)
#     y = a[i+3,:]
#     print(y)

print(train)
# [2879 * 3]
train_x = []
train_y = []
for i in range(len(train)-5):
    train_x.append(np.array(train.values[i:i+5, :].flatten()))
    train_y.append(np.array(train.values[i+5, :].flatten()))

train_x = np.array(train_x)
train_y = np.array(train_y)
# train_x [9*5, len(x)-5]
# train_y [9, len(x)-5]
# test_x [9*5]
print(train_x)
print(train_y)

# test_x = list(train_x[-1][3:])
# test_x.extend(train_y[-1])
# print(test_x)
test_x = train.values[-5:,:].reshape((1,-1))
print(test_x)


xs = tf.placeholder(tf.float32, [None, 15])
ys = tf.placeholder(tf.float32, [None, 3])

Weights_1 = tf.Variable(tf.random_normal([15, 54])*0.001)
biases_1 = tf.Variable(tf.zeros([1, 54]) + 0.001)
Wx_plus_b_1 = tf.matmul(xs, Weights_1) + biases_1

layer1_out = tf.nn.relu(Wx_plus_b_1)

Weights_2 = tf.Variable(tf.random_normal([54, 27])*0.001)
biases_2 = tf.Variable(tf.zeros([1, 27]) + 0.001)
Wx_plus_b_2 = tf.matmul(layer1_out, Weights_2) + biases_2

layer2_out = tf.nn.relu(Wx_plus_b_2)

Weights_3 = tf.Variable(tf.random_normal([27, 3])*0.001)
biases_3 = tf.Variable(tf.zeros([1, 3]) + 0.001)

prediction = tf.matmul(layer2_out, Weights_3) + biases_3
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate=0.05, beta1=0.9, beta2=0.999).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for step in range(0, 2000):
    sess.run(train_step, feed_dict={xs: train_x, ys: train_y})
    if step % 100 == 0:
        print(step, sess.run(loss, feed_dict={xs: train_x, ys: train_y}))

output = []
for i in range(2800):
    y_hat = sess.run(prediction, feed_dict={xs: test_x})
    # print(y_hat)
    output.append(y_hat)
    test_x = list(test_x[:, 3:].flatten())
    # print(test_x)
    test_x.extend(y_hat[0])
    test_x = np.array(test_x).reshape(1, -1)
    # print(test_x)

output = np.array(output).reshape((2800, 3))
print(output)
plt.plot(output)
plt.show()



# for j in range(9):
#     y_hat[j] = y_hat[j]*scale[1][j]+scale[0][j]
#
#
# y_hat = pd.DataFrame(y_hat)
# y_hat.to_excel('y_hat.xlsx')