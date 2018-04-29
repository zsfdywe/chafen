# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('ele_15min.csv', skip_blank_lines=False)
'''
def get_line(xn, yn):
    def line(x):
        index = -1
        # 找出x所在的区间
        for i in range(1, len(xn)):
            if x <= xn[i]:
                index = i - 1
                break
            else:
                i += 1
        if index == -1:
            return -100
        # 插值
        result = (x - xn[index + 1]) * yn[index] / float((xn[index] - xn[index + 1])) + (x - xn[index]) * yn[
            index + 1] / float((xn[index + 1] - xn[index]))
        return result
    return line
yn = [df[:-1]]
xn = [i for i in range(6306)]
# 分段线性插值函数
lin = get_line(xn, yn)
x = [i for i in range(6306)]
y = [lin(i) for i in x]
'''
y_hat = df.interpolate()
y_hat_b = y_hat[df.electricity.isnull()]
print(y_hat_b)
y_hat_b.to_csv('y_hat_l.csv')
df.plot()
y_hat.plot()
plt.show()