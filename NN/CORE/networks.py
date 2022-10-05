from random import sample

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

    def train(self, trainingData, solution, epochs, learningrate):
        samples = len(trainingData)
        for i in range(epochs):
            errorDisplay = 0
            for j in range(samples):
                output = self.forward(trainingData[j])

                errorDisplay += self.loss(solution[j], output)

                error = self.lossDerivative(solution[j], output)
                for layer in reversed(self.layers):
                    error = layer.backwardPropagation(error, learningrate)

            errorDisplay /= samples
            #print(f"training cycle {i} / {epochs}   error={errorDisplay}")
        print(f"training cycle done: {epochs}   average error={errorDisplay}")

    def predict(self, inputData):
        result = []
        for sample in inputData:
            output = sample
            result.append(self.forward(output))

        return result