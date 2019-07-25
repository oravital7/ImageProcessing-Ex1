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

     return np.dot(imRGB, yiq_from_rgb.T)

def transformYIQ2RGB(imYIQ:np.ndarray)->np.ndarray:
    rgb_from_yiq = np.array([[1,0.956,0.619],
                            [1,-0.272,-0.647],
                             [1,-1.106,1.703]])
    return np.dot(imYIQ, rgb_from_yiq.T)



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


def quantizeImage(imOrig:np.ndarray, nQuant:int, nIter:int):
    yiq_im = []

    if len(imOrig.shape) == 3:
        yiq_im = transformRGB2YIQ(imOrig)
        init_im = yiq_im[:, :, 0].copy()
    else:
        init_im = imOrig.copy()

    hist, bin_edges = np.histogram(init_im, 256)
    hist_cum = np.cumsum(hist)
    pixInSegment = int(hist_cum[-1] / nQuant)

    error, qArr, zArr = segmtents_quantization(bin_edges, hist, hist_cum, nIter, nQuant, pixInSegment)

    for segNum in range(nQuant):
        inSeg = np.logical_and(init_im >= zArr[segNum], init_im < zArr[segNum + 1])
        init_im[inSeg] = qArr[segNum]

    init_im[init_im == 1] = qArr[-1]

    im_quant = init_im
    if len(imOrig.shape) == 3:
        yiq_im[:, :, 0] = im_quant
        im_quant = transformYIQ2RGB(yiq_im)

    return [im_quant, np.array(error)]

def segmtents_quantization(bin_edges, hist, hist_cum, n_iter, n_quant, pixInSegment):
    zArr = np.array([0] + [bin_edges[np.where(hist_cum >= pixInSegment * i)[0][0]] for i in range(1, n_quant)] + [1])
    qArr = np.zeros(n_quant)
    error = []
    for i in range(n_iter):
        curZ = zArr.copy()
        curErr = 0

        for k in range(n_quant):
            if k != n_quant - 1:
                curSeg = np.intersect1d(np.where(bin_edges[:-1] >= zArr[k])[0],
                                        np.where(bin_edges[:-1] < zArr[k + 1])[0])
            else:
                curSeg = np.intersect1d(np.where(bin_edges[:-1] >= zArr[k])[0],
                                        np.where(bin_edges[:-1] <= zArr[k + 1])[0])

            qArr[k] = np.dot(bin_edges[curSeg], hist[curSeg]) / np.sum(hist[curSeg])
            curErr += np.dot(np.power(qArr[k] - bin_edges[curSeg], 2), hist[curSeg])
        error.append(curErr)
        zArr = np.array([0] + [(qArr[k] + qArr[k - 1]) / 2 for k in range(1, n_quant)] + [1])
        if np.array_equal(zArr, curZ):
            break
    return error, qArr, zArr