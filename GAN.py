from NN.CORE.layers import activation, avgPool, fullyConnected, maxPool, pooling
from NN.CORE.networks import network
import numpy as np
import os
from PIL import Image
from NN.UTIL.data import preprocess

def main():
    #rawData = "C:/Users/Kaan/Desktop/KI-Neural Network/GAN/DATA/data"
    processedData = "C:/Users/Kaan/Desktop/KI-Neural Network/GAN/DATA/processed"
    #preprocess(rawData, processedData)
    test = np.empty(shape=(len(os.listdir(processedData)), 3, 1000, 1000))
    i = 0
    for file in os.listdir(processedData):
        #print(i)
        test[i, :, :, :] = np.load(f"{processedData}\{file}")
        i = i+1
    GAN = network()
    GAN.addLayer(maxPool(test, 2, 2))
    i = 0
    for image in GAN.layers[0].forwardPropagation():
    #for image in test:
        image = image.result().astype(np.uint8)
        im = Image.fromarray(image.transpose(1,2,0).astype(np.uint8))
        im.save(f"your_file{i}.jpeg")
        i = i+1

if __name__ == "__main__":
    main()
    pass