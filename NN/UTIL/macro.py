def canStride(poolFactor, stride):
    if (poolFactor-1)%stride!=0:
        raise ValueError

def poolImgShape(shape, poolFactor, stride):
    return (shape[0], int(1 + (shape[1]-poolFactor)/stride), int(1 + (shape[2]-poolFactor)/stride))