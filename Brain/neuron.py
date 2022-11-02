import random

# https://medium.com/@sakeshpusuluri/activation-functions-and-weight-initialization-in-deep-learning-ebc326e62a5c

class neuron:
    def __init__(self, nbInput, actFunc, lossFunc) -> None:
        self.lr = 1 # learning rate
        self.bias = 1
        self.weight = [random.random() for _ in range(nbInput + 1)]
        self.actFunction = actFunc
        self.lossFunction = lossFunc
        pass

    #################### Train ####################
    def updateWeight(self, output: float, outputExpected: float, inputList: list[float]) -> None:
        error = self.lossFunction(outputExpected, output)
        for i in range(len(self.weight) -1):
            self.weight[i] += self.lr * error * inputList[i]
        self.weight[-1] += self.lr * error * self.bias

    def train(self, inputList: list[float], outputExpected: float) -> float:
        Z = self.bias * self.weight[-1]
        for i in range(len(inputList)):
            Z += inputList[i] * self.weight[i]
        output = self.actFunction(Z)
        self.updateWeight(output, outputExpected, inputList)

    #################### Test ####################
    def calc(self, inputList: list[float]) -> float:
        Z = self.bias * self.weight[-1]
        for i in range(len(inputList)):
            Z += inputList[i] * self.weight[i]
        output = self.actFunction(Z)
        return output
