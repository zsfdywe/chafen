# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as mpl

c_rho=[0,716,1934,1934,84,0]
dx=[0,0.025,0.06,0.06,0.05,0]
k=[9.3,0.19,1.4,1.4,0.042,9.3]
dt=180
period = 3600
method = "backward"

cap = [c_rho[i] * dx[i] * 1000 for i in range(1,5)]
r   = [dx[i] / k[i] for i in range(1,5)]
cap.insert(0, 0)
cap.append(0)
r.insert(0, 1/k[0])
r.append(1/k[-1])
ul  = [dt / 0.5 / (cap[i] + cap[i+1]) / r[i] for i in range(5)]
ur  = [dt / 0.5 / (cap[i] + cap[i+1]) / r[i+1] for i in range(5)]

def deepcopy(matrix):
    temp = []
    for i in range(len(matrix)):
        temp_temp = []
        for j in range(len(matrix)):
            temp_temp.append(matrix[i][j])
        temp.append(temp_temp)
    return temp

def cofactors(matrix,i,j):
    sub_matrix = deepcopy(matrix)
    sub_matrix.pop(i)
    for k in range(len(sub_matrix)):
        sub_matrix[k].pop(j)
    return sub_matrix

def determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    else:
        sum = 0
        for j in range(len(matrix)):
            sub_matrix = cofactors(matrix, 0, j)
            sum += matrix[0][j] * determinant(sub_matrix) * ((-1) ** (j))
        return sum

def matrix_inverse(matrix):
    det = determinant(matrix)
    matrix_star = []
    for i in range(len(matrix)):
        matrix_star_line = []
        for j in range(len(matrix)):
            matrix_star_line.append((-1)**(i+j) * determinant(cofactors(matrix,j,i))/det)  # i,j换位转置
        matrix_star.append(matrix_star_line)
    return matrix_star

def multi(m1,m2):
    return [[sum([m1[j][k] * m2[k][i] for k in range(len(m2))]) for i in range(len(m2[0]))] for j in range(len(m1))]

def reverse(matrix):
    return([[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))])

if method=="forward":
    temp = [20, 10, 10, 10, 10, 10, 10]
    output = []
    output.append(temp[1:6])
    for i in range(int(period/dt)):
        temp[1:6]=[round(ul[j]*temp[j] + (1-ul[j]-ur[j])*temp[j+1] + ur[j]*temp[j+2],3) for j in range(5)]
        output.append(temp[1:6])
        print(temp)
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
mpl.plot(output)
mpl.show()
print(output)