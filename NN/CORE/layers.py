from multiprocessing import pool
from NN import randomDropoutMatrix, avgPooling, maxPooling

from numpy import array, dot, random, reshape
import concurrent.futures

class layer(object):

    def __init__(self, *args, **kwargs):
        self.input = None
        self.output = None
        self.inputShape = None
        self.outputShape = None

    def forwardPropagation(self, data):
        pass

    def backwardPropagation(self, data, output_error, learning_rate):
        pass

class activation(layer):

    def __init__(self, activation, activationDerivative):
        super().__init__()
        self.activation = activation
        self.activationDerivative = activationDerivative

    def forwardPropagation(self, inputData):
        self.input = inputData
        self.output = self.activation(self.input)
        return self.output

    def backwardPropagation(self, error, learningRate):
        return self.activationDerivative(self.input) * error

class convolution(layer):#wip
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class dropout(layer):

    def __init__(self, dropoutRate, dropoutShape):
        super().__init__()
        self.dropoutShape = dropoutShape
        self.dropoutRate = dropoutRate
        self.dropoutMatrix = None
        self.dropoutCorrection = (1/(1-self.dropoutRate))

    def forwardPropagation(self, input):
        self.dropoutMatrix = randomDropoutMatrix(self.dropoutRate, self.dropoutShape)
        return input*self.dropoutCorrection*self.dropoutMatrix

    def backwardPropagation(self, input):
        return input*self.dropoutCorrection*self.dropoutMatrix

class flatten(layer):

    def __init__(self):
        super().__init__()

    def forwardPropagation(self, input):
        self.inputShape = input.shape
        return input.flatten()

    def backwardPropagation(self, input):
        return reshape(input, self.inputShape)

class fullyConnected(layer):

    def __init__(self, inputNeuronNumber, outputNeuronNumber):
        self.inputNeuronNumber = inputNeuronNumber
        self.outputNeuronNumber = outputNeuronNumber
        self.weights = random.rand(inputNeuronNumber, outputNeuronNumber)
        self.bias = random.rand(1, outputNeuronNumber)

    def forwardPropagation(self, inputData):
        self.input = inputData
        self.output = dot(self.input, self.weights) + self.bias
        return self.output

    def backwardPropagation(self, outputError, learningrate):
        inputError = dot(outputError, self.weights.T)
        weightsError = dot(self.input.T, outputError)
        self.weights -= learningrate * weightsError
        self.bias -= learningrate * outputError
        return inputError

class pooling(layer):

    def __init__(self, input, poolFactor, stride=2):
        super().__init__(input)
        self.input = input
        self.mode = None
        self.poolFactor = poolFactor
        self.stride = stride
    
    def forwardPropagation(self):
        self.inputShape = self.input.shape
        self.mode = avgPooling
        with concurrent.futures.ThreadPoolExecutor() as executor:
            pooledImages = [executor.submit(self.mode, image, self.poolFactor, self.stride) for image in self.input]
        
        for image in pooledImages:
            image = image.result()
        

        pooledImages = array(pooledImages)

        return pooledImages

    def backwardPropagation(self):
        raise NotImplementedError

class avgPool(pooling):
    def __init__(self, input, poolFactor, stride=2):
        super().__init__(input, poolFactor, stride)
        self.mode = avgPooling

class maxPool(pooling):
    def __init__(self, input, poolFactor, stride=2):
        super().__init__(input, poolFactor, stride)
        self.mode = maxPooling
