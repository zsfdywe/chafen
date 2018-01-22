# -*- coding: utf-8 -*-

##import matplotlib.pyplot as mpl

##初期条件を入力する
c_rho=[0,716,1934,1934,84,0]
dx=[0,0.025,0.06,0.06,0.05,0]
k=[9.3,0.19,1.4,1.4,0.042,9.3]
dt=180
period = 36000
method = "backward"

##熱容量cap、熱抵抗ｒとur、ulの計算
cap = [c_rho[i] * dx[i] * 1000 for i in range(1,5)]
r   = [dx[i] / k[i] for i in range(1,5)]
cap.insert(0, 0)
cap.append(0)
r.insert(0, 1/k[0])
r.append(1/k[-1])
ul  = [dt / 0.5 / (cap[i] + cap[i+1]) / r[i] for i in range(5)]
ur  = [dt / 0.5 / (cap[i] + cap[i+1]) / r[i+1] for i in range(5)]

##逆行列の計算
#余因子を計算する
def cofactors(matrix,i,j):
    def deepcopy(matrix):
        temp = []
        for i in range(len(matrix)):
            temp_temp = []
            for j in range(len(matrix)):
                temp_temp.append(matrix[i][j])
            temp.append(temp_temp)
        return temp
    sub_matrix = deepcopy(matrix)
    sub_matrix.pop(i)
    for k in range(len(sub_matrix)):
        sub_matrix[k].pop(j)
    return sub_matrix

#行列式を計算する
def determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    else:
        sum = 0
        for j in range(len(matrix)):
            sub_matrix = cofactors(matrix, 0, j)
            sum += matrix[0][j] * determinant(sub_matrix) * ((-1) ** (j))
        return sum

#逆行列を計算する
def matrix_inverse(matrix):
    det = determinant(matrix)
    matrix_star = []
    for i in range(len(matrix)):
        matrix_star_line = []
        for j in range(len(matrix)):
            matrix_star_line.append((-1)**(i+j) * determinant(cofactors(matrix,j,i))/det)  # i,jの位置転換
        matrix_star.append(matrix_star_line)
    return matrix_star

##転置行列を行列する
def reverse(matrix):
    return([[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))])

##行列の内積を計算する
def multi(m1,m2):
    return [[sum([m1[j][k] * m2[k][i] for k in range(len(m2))]) for i in range(len(m2[0]))] for j in range(len(m1))]

##前進差分と後退差分により温度分布を計算する
if method=="forward":
    temp = [20, 10, 10, 10, 10, 10, 10]
    output = []
    for i in range(int(period/dt)):
        temp[1:6]=[ul[j]*temp[j] + (1-ul[j]-ur[j])*temp[j+1] + ur[j]*temp[j+2] for j in range(5)]
        output.append(temp.copy())
elif method=="backward":
    temp = [20, 10, 10, 10, 10, 10, 10]
    output=[]
    for i in range(int(period/dt)):
        U = [[1+ul[0]+ur[0], -ur[0], 0, 0, 0], [-ul[1], 1+ul[1]+ur[1], -ur[1], 0, 0],
             [0, -ul[2], 1+ul[2]+ur[2], -ur[2], 0], [0, 0, -ul[3], 1+ul[3]+ur[3], -ur[3]],
             [0, 0, 0, -ul[4], 1+ul[4]+ur[4]]]
        U_inverse = matrix_inverse(U)
        temp[1] += ul[0]*temp[0]
        temp[5] += ur[4] * temp[6]
        temp[1:6]=reverse(multi(U_inverse, reverse([temp[1:6]])))[0]
        output.append(temp.copy())
##mpl.plot(output)
##mpl.show()

for i in output:
    temp = []
    for j in i:
        temp.append(round(j,3))
    print(temp[1:6])