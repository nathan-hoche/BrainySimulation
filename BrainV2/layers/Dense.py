import numpy as np

class Dense():
    def __init__(self, actFunc, gradient, nbNeuron, nbInput, learningRate, isFirst) -> None:
        self.isFirst = isFirst
        self.learningRate = learningRate
        self.actFunc = actFunc
        self.gradient = gradient
        self.weights = np.random.uniform(size=(nbInput, nbNeuron)) 
        self.bias = np.random.uniform(size=(1 ,nbNeuron))
        self.output = None
        self.input = None
        pass

    def __str__(self) -> str:
        return "Dense Layer: " + str(self.weights.shape[1]) + " neurons, " + str(self.weights.shape[0]) + " inputs"

    def size(self) -> int:
        return self.weights.shape[1]

    ## Train ##

    def updateWeight(self, outputExpected:float, velocity:float=None) -> None:
        if (velocity is None):
            velocity = (outputExpected - self.output)
        gradient = self.gradient(velocity, self.output, gradient=self.actFunc)
        # print("\ngradient", gradient)
        # print("\nweights", self.weights)
        if (not self.isFirst):
            newVelocity = gradient * self.weights.T
        else:
            newVelocity = None
        # print("\nnewVelocity", newVelocity, "\n----")
        # # print(gradient.shape)
        # t = self.input
        self.weights += self.learningRate * np.matmul(np.transpose(self.input), gradient)
        self.bias += np.sum(self.learningRate * gradient, axis=0)
        return newVelocity

    ## Evaluate ##

    def calc(self, inputList: list[float]) -> list[float]|float:
        self.input = np.array(inputList)
        self.output = self.actFunc(np.dot(inputList, self.weights) + self.bias)
        return self.output

