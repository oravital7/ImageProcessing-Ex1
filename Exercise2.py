import numpy as np

import math
try:
    from cv2 import cv2
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
           # print("i: ",i)
            if i-k+j >= 0 and i-k+j < sizeSignal:
                if i == 1:
                     print('inSignal : %d kernel1 : %d',inSignal[i-k+j],kernel1[sizeKernel-1-j])
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
    rows = cv2.filter2D(inImage, -1, kernel)
    cols = cv2.filter2D(inImage, -1 , np.transpose(kernel))
    rows = rows*rows
    cols = cols*cols
    power = rows*rows+cols*cols

    for i in range(0, power.shape[0]):
        for j in range(0, power.shape[1]):
            power[i,j] = power[i,j]**(0.5)

    cv2.imshow("myImage", power)
    cv2.waitKey(0)



 #2.2
def blurImage1(inImage:np.ndarray,kernelSize:np.ndarray)->np.ndarray:
    kernel = CreateBlurKernel(kernelSize)
    #rs = conv2D(inImage, kernel)
    rs=cv2.filter2D(inImage, -1, kernel)
    cv2.imshow("myImage", rs)
    cv2.waitKey(0)
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




 #3.0
def blurImage2(inImage:np.ndarray,kernelSize:np.ndarray)->np.ndarray:
    blur = cv2.GaussianBlur(inImage, (kernelSize, kernelSize), 0)
    return blur


#3.1
def edgeDetectionSobel(I:np.ndarray)->(np.ndarray,np.ndarray):
    Gx = np.array([[-1,0,1],
                  [-2,0,2],
                  [-1,0,1]])
    Gy = np.array([[-1,-2,-1],
                  [0,0,0],
                  [1,2,1]])


    Gcx = cv2.filter2D(I,-1, Gx)
    Gcy =  cv2.filter2D(I,-1, Gy)



    rs = np.matmul(Gcx,Gcx)+np.matmul(Gcy,Gcy)

    rs=np.sqrt(rs)
    return rs


#3.2
def edgeDetectionZeroCrossingSimple(I:np.ndarray)->(np.ndarray,np.ndarray):

    src_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(src_gray, cv2.CV_16S, ksize=7)

    # TO DO :
    # Look for patterns like {+, 0, -} or {+, -} (zerocrossing) not sure what to do

    return lap

#3.3
def edgeDetectionZeroCrossingLOG(I:np.ndarray)->(np.ndarray,np.ndarray):
    # smooth with gaussian
    I = blurImage2(I, 9)

    # turn to GrayScale and activate laplace filter
    src_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(src_gray, cv2.CV_16S, ksize=7)


    # show the img
    # activate abs only if you want to show the picture
    # make sure you find the edge points before using abs

    lap = cv2.convertScaleAbs(lap)

    return lap


#3.3
def edgeDetectionCanny(I:np.ndarray)->(np.ndarray,np.ndarray):
    #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
    #first blur the image
    I=blurImage2(I,5)

    #claculate magnitude and angle
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    Gcx = cv2.filter2D(I, -1, Gx)
    Gcy = cv2.filter2D(I, -1, Gy)

    rs = np.matmul(Gcx, Gcx) + np.matmul(Gcy, Gcy)

    rs = np.sqrt(rs)
    angle = np.arctan(Gcx/Gcy)

    #need to check for each pixel , go through his neighbors and check if he is maximum
    #if he is he stays otherwise change it to zero

 # 4.0
def houghCircle(I:np.ndarray,minRadius:float,maxRadius:float)->np.ndarray:
    #https://www.youtube.com/watch?v=-o9jNEbR5P8
    I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    I=blurImage1(I,21)

    circ= cv2.HoughCircles(I,cv2.CV_HOUGH_GRADIENT,0.9,120
                          ,param1=50,param2=30
                          ,minRadius=minRadius ,maxRadius=maxRadius)
    roundCirc=np.uint16(np.around(circ))

    return roundCirc


