import numpy as np
from BrainV1.utils import Gradient

# https://medium.com/@sakeshpusuluri/activation-functions-and-weight-initialization-in-deep-learning-ebc326e62a5c

class neuron:
    def __init__(self, nbInput, actFunc, lossFunc, learningRate=1.0) -> None:
        self.lr = learningRate # learning rate
        self.bias = 1
        self.weight = np.random.uniform(-1, 1, nbInput)
        self.actFunction = actFunc
        self.gradient = lossFunc
        self.isGradientDescent = False
        if (self.gradient == Gradient.basic or self.gradient == Gradient.hebbian or self.gradient == Gradient.oja):
            self.isGradientDescent = True

    #################### Train ####################
    def updateWeight(self, output: float, outputExpected: float, inputList: list[float], velocity) -> None:
        for i in range(len(self.weight)):
            tmp, v = self.gradient(outputExpected, output, inputList[i], self.lr, velocity=velocity, gradient=self.actFunction)
            self.weight[i] += tmp
        if (self.isGradientDescent):
            self.bias += self.gradient(outputExpected, output, 1, self.lr)[0]
        else:
            self.bias += np.sum(self.lr * v, axis=0)
        return v

    def train(self, inputList: list[float], outputExpected: float) -> float:
        Z = np.dot(inputList, self.weight) + self.bias
        output = self.actFunction(Z)
        self.updateWeight(output, outputExpected, inputList)

    #################### Test ####################
    def calc(self, inputList: list[float]) -> float:
        Z = np.dot(inputList, self.weight) + self.bias
        return self.actFunction(Z)

    def printWeight(self) -> None:
        print("Weight:", self.weight, "Bias:", self.bias)
