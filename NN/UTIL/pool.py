import numpy as np
from NN.UTIL.macro import canStride, poolImgShape
import concurrent.futures

def avgPooling(input, poolFactor, stride):
    #canStride(poolFactor, stride)
    pooledImg = np.zeros(poolImgShape(input.shape, poolFactor, stride))
    with concurrent.futures.ThreadPoolExecutor() as poolOperator:
        image = [poolOperator.submit(avgPoolOperation, imglayer, pooledImg.shape, poolFactor, stride) for imglayer in input]

    retImg = np.empty_like(image)

    for imgLayer in image:
        retImg = np.array(imgLayer.result())

    return np.array(retImg)

def maxPooling(input, poolFactor, stride):
    #canStride(poolFactor, stride)
    pooledImg = np.zeros(poolImgShape(input.shape, poolFactor, stride))
    with concurrent.futures.ThreadPoolExecutor() as poolOperator:
        results = [poolOperator.submit(maxPoolOperation, input, pooledImg, poolFactor, stride) for imglayer in input]
    
    for result in results:
        result.result()
    return results

def avgPoolOperation(imgSlice, pooledImgShape, poolFactor, stride):
    i = 0
    j = 0
    #print(f"({i*stride}:{i*stride+(poolFactor-1)}, {j*stride}:{j*stride+(poolFactor-1)})")
    pooledImgSlice = np.zeros(pooledImgShape)
    while i < pooledImgShape[0]: #in range(0, imgSlice.shape[0], stride):
        while j < pooledImgShape[1]:#in range(0, imgSlice.shape[2, stride]):
            pooledSlice = imgSlice[(i*stride):(i*stride+(poolFactor-1)), (j*stride):(j*stride+(poolFactor-1))]
            #print(f"###{i*stride}:{i*stride+(poolFactor-1)}###{j*stride}:{j*stride+(poolFactor-1)}###\n")
            pooledImgSlice[i, j] = np.average(pooledSlice)
            j = j+1
        i = i+1
    #print(pooledImgShape)
    return pooledImgSlice

def maxPoolOperation(imgSlice, pooledImg, poolFactor, stride):
    i = 0
    j = 0
    while i <= imgSlice.shape[0]: #in range(0, imgSlice.shape[0], stride):
        try:
            while j <= imgSlice.shape[1]:#in range(0, imgSlice.shape[2, stride]):
                try:
                    pooledImg[i, j] = np.maximum(imgSlice[i*stride:i*stride+(poolFactor-1), j*stride:j*stride+(poolFactor-1)])
                    j = j+stride
                except:
                    continue
            i = i+stride
        except:
            continue
    return pooledImg