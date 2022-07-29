from pathlib import Path

class network:

    def __init__(self):
        self.data = None
        self.dataTensor = None
        self.layers = []
        self.loss = None
        self.lossDerivative = None

    def addLayer(self, layer):
        self.layers.append(layer)

    def useLoss(self, loss, lossDerivative):
        self.loss = loss
        self.lossDerivative = lossDerivative

    def forward(self, inputTrainingData):
        output = inputTrainingData
        for layer in self.layers:
            output = layer.forwardPropagation(output)
        return output

    def backward(self, dataExpectation, learningrate, output, error=None):
        if error == None:
            error = self.lossDerivative(dataExpectation, output)
        for layer in reversed(self.layers):
            error = layer.backwardPropagation(error, learningrate)
        return error