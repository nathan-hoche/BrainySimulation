import numpy as np

class neuron:
    def __init__(self, nbInput, actFunc, lossFunc, learningRate=1.0) -> None:
        self.lr = learningRate # learning rate
        self.bias = 1
        self.weight = np.random.uniform(-1, 1, nbInput)
        self.actFunction = actFunc
        self.gradient = lossFunc
        pass

    #################### Train ####################
    def updateWeight(self, output: float, outputExpected: float, inputList: list[float]) -> None:
        for i in range(len(self.weight)):
            self.weight[i] += self.lr * self.gradient(outputExpected, output, inputList[i])
        self.bias += self.lr * self.gradient(outputExpected, output, 1)

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