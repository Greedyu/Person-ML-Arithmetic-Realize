import numpy as np;
from tensorflow import sigmoid

def gradAscent(dataMatIn , classLabels):
    dataMatrix = np.mat(dataMatIn)  #转换成numpy 的mat
    labelMat = np.mat(classLabels).transpose()  #转换成numpy的mat格式，再装置
    m,n = np.shape(dataMatrix);  #获取dataMatrix的行列数
    alpha = 0.01     #更新速率
    maxCycles = 5000   #迭代次数
    weights = np.ones((n,1))    #初始化权重矩阵
    for i in range(maxCycles):
        h = sigmoid(dataMatrix * weights)    # 计算预测值h
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error    # 我们的梯度上升公式！
    return weights.getA()                       #将矩阵转换为数组，返回权重数组
