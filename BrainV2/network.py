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

    #### LAYERS ####

    def addNewLayer(self, actFunc, gradient, nbNeuron, nbInput=None) -> None:
        if nbInput is None:
            nbInput = self.layers[-1].size()
        newLayer = Dense(actFunc, gradient, nbNeuron, nbInput, self.learningRate, self.isFirst)
        self.layers.append(newLayer)
        self.isFirst = False

    #### CALCULATION ####

    def calc(self, inputList: list[float]) -> list[float]:
        output = inputList
        for layer in self.layers:
            output = layer.calc(output)
        return output

    #### TRAINING ####

    def train(self, inputList: list[float], outputExpected: float|list[float], epochs:int=100) -> None:
        loss = None
        print("epochs:", epochs)
        for i in range(epochs):
            output = self.calc(inputList)
            if (self.lossFunc is not None):
                loss = self.lossFunc(output, outputExpected)
                self.losses.append(loss)
            velocity = None
            for layer in reversed(self.layers):
                velocity = layer.updateWeight(outputExpected, velocity)
    
    #### INFORMATION ####
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