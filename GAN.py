from NN.CORE.networks import network
from NN.UTIL.data import preprocess

def main():
    rawData = "C:/Users/Kaan/Desktop/KI-Neural Network/GAN/DATA/data"
    processedData = "C:/Users/Kaan/Desktop/KI-Neural Network/GAN/DATA/processed"
    preprocess(rawData, processedData)
    GAN = network()

if __name__ == "__main__":
    main()