from threading import activeCount
from NN.CORE.layers import activation, avgPool, fullyConnected, maxPool, pooling
from NN.CORE.networks import network
from NN.UTIL.data import preprocess
from NN.UTIL.profiling import profile
from NN.UTIL.activation import tanh, tanhDerivative
from NN.UTIL.loss import mse, mseDerivative
import numpy as np
import os
from PIL import Image

@profile
def main():
    #rawData = "C:/Users/Kaan/Desktop/KI-Neural Network/GAN/DATA/data"
    #processedData = "C:/Users/Kaan/Desktop/KI-Neural Network/GAN/DATA/processed"
    #preprocess(rawData, processedData)
    #test = np.empty(shape=(len(os.listdir(processedData)), 3, 1000, 1000))
    #i = 0
    #for file in os.listdir(processedData):
    #    test[i, :, :, :] = np.load(f"{processedData}\{file}")
    #    i = i+1
    GAN = network()
    trainingData = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    dataLabel = np.array([[[0]], [[1]], [[1]], [[0]]])

    GAN.addLayer(fullyConnected(2, 3))
    GAN.addLayer(activation(tanh, tanhDerivative))
    GAN.addLayer(fullyConnected(3, 1))
    GAN.addLayer(activation(tanh, tanhDerivative))

    GAN.useLoss(mse, mseDerivative)

    GAN.train(trainingData, dataLabel, 5000, .1)

    print(GAN.predict(trainingData))
    #i = 0
    #for image in GAN.layers[0].forwardPropagation():
    #for image in test:
    #    image = image.result().astype(np.uint8)
    #    im = Image.fromarray(image.transpose(1,2,0).astype(np.uint8))
    #    im.save(f"your_file{i}.jpeg")
    #    i = i+1

if __name__ == "__main__":
    main()
    pass