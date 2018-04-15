import numpy as np
import pandas as pd
import tensorflow as tf

train = pd.DataFrame(pd.read_excel('train.xlsx'))
test = pd.DataFrame(pd.read_excel('test.xlsx'))

train['doy'] = train.index.dayofyear
train['hr'] = [train.index.hour[i] + train.index.minute[i]/60 for i in range(len(train.index.hour))]
train.insert(0, 'dayofyear', train['doy'])
train.insert(1, 'hour', train['hr'])
del train['doy']
del train['hr']

test['doy'] = test.index.dayofyear
test['hr'] = [test.index.hour[i] + test.index.minute[i]/60 for i in range(len(test.index.hour))]
test.insert(0, 'dayofyear', test['doy'])
test.insert(1, 'hour', test['hr'])
del test['doy']
del test['hr']
# train['hoursofyear'] = (np.array((train.index - train.index[0]).view('int64') / 3.6e12))


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
scale0 = scale[:, -1]
scale = scale[:, 0:-1]
test_std = standardise(test.values, scale)

train = pd.DataFrame(train_std)
test = pd.DataFrame(test_std)

train.to_excel('train_std.xlsx')
test.to_excel('test_std.xlsx')

train_x = np.array(train.values[:, 0:4])
train_y = np.array(train.values[:, -1]).reshape((-1, 1))

xs = tf.placeholder(tf.float32, [None, 4])
ys = tf.placeholder(tf.float32, [None, 1])

Weights_1 = tf.Variable(tf.random_normal([4, 12])*0.01)
biases_1 = tf.Variable(tf.zeros([1, 12]) + 0.01)
Wx_plus_b_1 = tf.matmul(xs, Weights_1) + biases_1

layer1_out = tf.nn.relu(Wx_plus_b_1)

Weights_2 = tf.Variable(tf.random_normal([12, 1])*0.01)
biases_2 = tf.Variable(tf.zeros([1, 1]) + 0.01)

prediction = tf.matmul(layer1_out, Weights_2) + biases_2
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate=0.05, beta1=0.9, beta2=0.999).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for step in range(0, 5000):
    sess.run(train_step, feed_dict={xs:train_x, ys:train_y})
    if step % 500 == 0:
        print(step, sess.run(loss, feed_dict={xs:train_x, ys:train_y}))
y_hat = sess.run(prediction, feed_dict={xs: test.values})


y_hat = y_hat*scale0[1]+scale0[0]
y_hat = pd.DataFrame(y_hat)

y_hat.to_excel('y_hat.xlsx')



