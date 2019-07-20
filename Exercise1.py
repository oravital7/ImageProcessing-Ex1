import numpy as np
import math
try:
    from cv2 import cv2 as cv
except ImportError:
    pass

np.set_printoptions(threshold=np.inf)

def imReadAndConvert(filename:str, representation:int)->np.ndarray:
    if representation == 1:
        img = cv.imread(filename,0)
    elif representation == 2:
        img = cv.imread(filename,1)

    else:
        raise ValueError('Value not expected.')

    return img/255.0


def imDisplay(filename:str, representation:int):
    img = imReadAndConvert(filename,representation)
    cv.imshow("img",img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def transformRGB2YIQ(imRGB:np.ndarray)->np.ndarray:
     yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                                [0.59590059, -0.27455667, -0.32134392],
                                [0.21153661, -0.52273617, 0.31119955]])

     return np.dot(imRGB, yiq_from_rgb.T.copy())

def transformYIQ2RGB(imYIQ:np.ndarray)->np.ndarray:
    rgb_from_yiq = np.array([[1,0.956,0.619],
                            [1,-0.272,-0.647],
                             [1,-1.106,1.703]])
    return np.dot(imYIQ, rgb_from_yiq.T.copy())



def histogramEqualize(imOrig:np.ndarray)->(np.ndarray,np.ndarray,np.ndarray):
    imOrig *= 255
    imOrig = imOrig.astype(int)
    img1 = imOrig

    height = imOrig.shape[0]
    width = imOrig.shape[1]
    pixels = width * height

    if(not isGrayScale(imOrig)):
        img1 = transformRGB2YIQ(imOrig)[:,:,0]
        img1 = img1.astype(int)


    hist_origi = histogram(img1)
    cum_hist = cumulative_histogram(hist_origi)

    for i in np.arange(height):
        for j in np.arange(width):
            a = img1.item(i, j)
            b = math.floor(cum_hist[a] * 255.0 / pixels)
            img1.itemset((i, j), b)

    hist_eq = histogram(img1)
    return (imOrig, hist_origi,hist_eq)



def histogram(img):
    height = img.shape[0]
    width = img.shape[1]

    hist = np.zeros((256))

    for i in np.arange(height):
        for j in np.arange(width):
            a = img.item(i, j)
            hist[a] += 1

    return hist


def cumulative_histogram(hist):
    cum_hist = hist.copy()

    for i in np.arange(1, 256):
        cum_hist[i] = cum_hist[i - 1] + cum_hist[i]

    return cum_hist

def isGrayScale(img):
    if(len(img.shape) < 3): return True
    else: return False

# imDisplay("/home/oravital7/Downloads/rgbImage.png",1)
# transformYIQ2RGB(imReadAndConvert("/home/oravital7/Downloads/rgbImage.png",2))
x = imReadAndConvert("/home/oravital7/Downloads/rgbImage.png",1)
y = histogramEqualize(x)
print(y[2])
# print(y)
