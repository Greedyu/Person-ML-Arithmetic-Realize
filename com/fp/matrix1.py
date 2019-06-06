#主要处理大量数字的应用
import numpy
import numpy as np

m = numpy.mat([2,3])
n = numpy.mat([[4],[5]])
x = np.array([[1,2],[3,4]])

# print(m*n)
# print(x.shape)
# print(x[:,1])
# print(m*x)

X = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0]]).T
print('X = \n%s\n' % X)

X_mean = np.mean(X, 1)
E = np.zeros([len(X), len(X)])
for i in range(len(X)):
    for j in range(i, len(X)):
        E[j, i] = E[i, j] = (X[i] - X_mean[i]).dot(X[j] - X_mean[j]) / len(X[i])
print('E = \n%s\n' % E)

print("np.cov(X, bias=1) = \n%s\n" % np.cov(X, bias=1))
print("np.cov(X) = \n%s\n" % np.cov(X))