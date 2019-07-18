import numpy as np
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


# imDisplay("/home/oravital7/Downloads/rgbImage.png",2)
transformYIQ2RGB(imReadAndConvert("/home/oravital7/Downloads/rgbImage.png",2))