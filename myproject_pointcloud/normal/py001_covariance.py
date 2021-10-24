import numpy as np
np.random.seed(10086)

def de_mean(x):
    xmean = np.mean(x)
    return [xi - xmean for xi in x]

def covariance(x, y):
    n = len(x)
    return np.dot(de_mean(x), de_mean(y)) / (n-1)

MySample = np.random.rand(10, 3)*50
MySample = np.rint(MySample)
MySample = MySample.astype(np.int8)
print(MySample)

dim1 = MySample[:,0]
dim2 = MySample[:,1]
dim3 = MySample[:,2]
cov12 = covariance(dim1, dim2)
cov13 = covariance(dim1, dim3)
cov23 = covariance(dim2, dim3)

var1 = np.var(dim1)
var2 = np.var(dim2)
var3 = np.var(dim3)
tvar1 = np.sum((dim1 - np.mean(dim1)) ** 2) / (len(dim1)-1)
tvar2 = np.sum((dim2 - np.mean(dim2)) ** 2) / (len(dim2)-1)
tvar3 = np.sum((dim3 - np.mean(dim3)) ** 2) / (len(dim3)-1)

print(
np.array(
    [
        [var1, cov12, cov13],
        [cov12, var2, cov23],
        [cov13, cov23, var3],
    ]
)
)

print(
np.array(
    [
        [tvar1, cov12, cov13],
        [cov12, tvar2, cov23],
        [cov13, cov23, tvar3],
    ]
)
)

np_covariance = np.cov(MySample, rowvar=False)
print(np_covariance)