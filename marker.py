import numpy as np
import cv2

dims = 2200
def convert2Int(x):
    x_ = x.split()
    a = int(float(x_[0])*1000)
    b = int(float(x_[1])*1000)
    return a, b

matrix = np.zeros((dims, dims))

f = open(r"C:/Users/xuanhieuktsx/data_marker/data/vl/1670569183.txt")
data = f.readlines()
for i in data[1:]:
    a , b = convert2Int(i)
    a += 1100
    b += 1100
    matrix[a][b] = 255
f.close()

matrix = cv2.resize(matrix, (800,800))
cv2.imshow("im", matrix)
cv2.waitKey(0)
