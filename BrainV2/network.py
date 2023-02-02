from BrainV2.layers.Dense import Dense
import matplotlib.pyplot as plt

class network():
    def __init__(self, learningRate=1.0, lossFunc=None) -> None:
        self.layers = []
        self.lossFunc = lossFunc
        self.learningRate = learningRate
        self.isFirst = True
        self.losses = []
        pass

    def addNewLayer(self, actFunc, gradient, nbNeuron, nbInput=None) -> None:
        if nbInput is None:
            nbInput = self.layers[-1].size()
        newLayer = Dense(actFunc, gradient, nbNeuron, nbInput, self.learningRate, self.isFirst)
        self.layers.append(newLayer)
        self.isFirst = False

    def calc(self, inputList: list[float]) -> list[float]:
        output = inputList
        for layer in self.layers:
            output = layer.calc(output)
        return output

    def train(self, inputList: list[float], outputExpected: float|list[float]) -> None:
        output = self.calc(inputList)
        if (self.lossFunc is not None):
            self.losses.append(self.lossFunc(output, outputExpected))
        velocity = None
        for layer in reversed(self.layers):
            velocity = layer.updateWeight(outputExpected, velocity)
    
    def printLayers(self):
        print("Network:")
        x = 0
        for layer in self.layers:
            print("Layer", x, ":", layer)
            x += 1
    
    def displayLoss(self):
        if (self.lossFunc is None):
            print("No loss function setted.")
            return
        plt.plot(self.losses)
        plt.show()