###TODO: make account for stride in PoolOperation

import numpy as np
from NN.UTIL.macro import canStride, poolImgShape
import concurrent.futures

def avgPooling(input, poolFactor, stride):
    #canStride(poolFactor, stride)
    pooledImg = np.zeros(poolImgShape(input.shape, poolFactor, stride))
    with concurrent.futures.ThreadPoolExecutor() as poolOperator:
        image = [poolOperator.submit(PoolOperation, imglayer, pooledImg.shape, poolFactor, stride, np.average) for imglayer in input]

    for imageLayer in range(len(image)):
        pooledImg[imageLayer] = image[imageLayer].result()

    return np.array(pooledImg)

def maxPooling(input, poolFactor, stride):
    #canStride(poolFactor, stride)
    pooledImg = np.zeros(poolImgShape(input.shape, poolFactor, stride))
    with concurrent.futures.ThreadPoolExecutor() as poolOperator:
        image = [poolOperator.submit(PoolOperation, imglayer, pooledImg.shape, poolFactor, stride, np.max) for imglayer in input]

    for imageLayer in range(len(image)):
        pooledImg[imageLayer] = image[imageLayer].result()

    return np.array(pooledImg)

def PoolOperation(imgSlice, pooledImgShape, poolFactor, stride, mode):
    pooledImgSlice = np.zeros(shape=(pooledImgShape[1], pooledImgShape[2]))
    for i in range(pooledImgShape[1]):
        for j in range(pooledImgShape[2]):
            pooledImgSlice[i, j] = mode(imgSlice[i*poolFactor:i*poolFactor+poolFactor, j*poolFactor:j*poolFactor+poolFactor])
    return pooledImgSlice