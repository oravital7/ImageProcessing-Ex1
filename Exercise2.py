import numpy as np

import math
try:
    from cv2 import cv2 as cv
except ImportError:
    pass

np.set_printoptions(threshold=np.inf)


# 2.0
def conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray:

    if len(inSignal) < len(kernel1):
        temp = inSignal
        inSignal = kernel1
        kernel1 = temp

    sizeSignal = len(inSignal)
    sizeKernel = len(kernel1)

    k = sizeKernel-2
    if sizeKernel == 2: k=k+1

    result = np.zeros(sizeSignal)

    for i in range(0 , sizeSignal):
        sums = 0
        for j in range(0, sizeKernel):
            print("i: ",i)
            if i-k+j >= 0 and i-k+j < sizeSignal:
             sums += inSignal[i-k+j]*kernel1[sizeKernel-1-j]
        result[i] = sums

    return result

def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray:
    h = inImage.shape[0]
    w = inImage.shape[1]

    result = inImage.copy()
    for i in range(0, h):
        for j in range(0, w):
            sums = inImage[i, j] * kernel2[1, 1]
            if i != 0: sums += inImage[i - 1, j] * kernel2[0, 1]
            if i != 0 and j != 0: sums += inImage[i - 1, j - 1] * kernel2[0, 0]
            if j != 0: sums += inImage[i, j - 1] * kernel2[1, 0]
            if i != 0 and j != w - 1: sums += inImage[i - 1, j + 1] * kernel2[0, 2]
            if j != w - 1: sums += inImage[i, j + 1] * kernel2[1, 2]
            if i != h - 1 and j != 0: sums += inImage[i + 1, j - 1] * kernel2[2, 0]
            if i != h - 1: sums += inImage[i + 1, j] * kernel2[2, 1]
            if i != h - 1 and j != w - 1: sums += inImage[i + 1, j + 1] * kernel2[2, 2]
            result[i, j] = sums

    return result

#2.1
def convDerivative(inImage:np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    rows = cv.filter2D(img, -1, kernel)
    cols = cv.filter2D(img, -1 , np.transpose(kernel))
    rows = rows*rows
    cols = cols*cols
    power = rows*rows+cols*cols

    for i in range(0, power.shape[0]):
        for j in range(0, power.shape[1]):
            power[i,j] = power[i,j]**(0.5)

    cv.imshow("myImage", power)
    cv.waitKey(0)



 #2.2
def blurImage1(inImage:np.ndarray,kernelSize:np.ndarray)->np.ndarray:
    kernel = CreateBlurKernel(kernelSize)

    rs = conv2D(inImage, kernel)
    print("im")
    cv.imshow("myImage", rs)
    cv.waitKey(0)
    return rs




def CreateBlurKernel(kernelSize):
    kernel = np.zeros(shape=(kernelSize,kernelSize),dtype=np.float)
    sum = 0
    for u in range(kernelSize):
        for v in range(kernelSize):
            uc = u - (kernelSize - 1) / 2;
            vc = v - (kernelSize - 1) / 2;
            g = math.exp(-(uc * uc + vc * vc) / (2));
            sum += g;
            kernel[u][v] = g;

    for u in range(kernelSize):
        for v in range(kernelSize):
            kernel[u][v] /= sum

    return kernel




 #2.2
def blurImage2(inImage:np.ndarray,kernelSize:np.ndarray)->np.ndarray:
    blur = cv.GaussianBlur(inImage, (kernelSize, kernelSize), 0)
    return blur


#2.3
def edgeDetectionSobel(I:np.ndarray)->(np.ndarray,np.ndarray):
    Gx = np.array([[-1,0,1],
                  [-2,0,2],
                  [-1,0,1]])
    Gy = np.array([[-1,-2,-1],
                  [0,0,0],
                  [1,2,1]])

    Gcx = cv.filter2D(I,-1, Gx)
    Gcy =  cv.filter2D(I,-1, Gy)

    rs = np.matmul(Gcx,Gcx)+np.matmul(Gcy,Gcy)

    cv.imshow("myImage", rs)
    cv.waitKey(0)

img = cv.imread("/home/oravital7/PycharmProjects/untitled/testSmooth.jpeg",0)
edgeDetectionSobel(img)
