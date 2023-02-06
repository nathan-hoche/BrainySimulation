from BrainV2.layers.Dense import Dense
from BrainV2.layers.Flatten import Flatten
from BrainV2.layers.MaxPool1D import MaxPool1D
from BrainV2.layers.MaxPool2D import MaxPool2D
from BrainV2.layers.AveragePool1D import AveragePool1D
from BrainV2.layers.AveragePool2D import AveragePool2D
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

    def addDenseLayer(self, actFunc, gradient, nbNeuron, nbInput=None) -> None:
        if nbInput is None:
            nbInput = self.layers[-1].size()
        newLayer = Dense(actFunc, gradient, nbNeuron, nbInput, self.learningRate, self.isFirst)
        self.layers.append(newLayer)
        self.isFirst = False
    
    def addFlattenLayer(self, nbInput=None) -> None:
        if nbInput is None:
            nbInput = self.layers[-1].size()
        self.layers.append(Flatten(nbInput))
    
    def addMaxPool2DLayer(self, pool_size=(2, 2), strides=None, padding=False, nbInput=None) -> None:
        if nbInput is None:
            nbInput = self.layers[-1].size()
        self.layers.append(MaxPool2D(nbInput, pool_size, strides, padding))
    
    def addMaxPool1DLayer(self, pool_size=2, strides=None, padding=False, nbInput=None) -> None:
        if nbInput is None:
            nbInput = self.layers[-1].size()
        self.layers.append(MaxPool1D(nbInput, pool_size, strides, padding))
    
    def addAveragePool2DLayer(self, pool_size=(2, 2), strides=None, padding=False, nbInput=None) -> None:
        if nbInput is None:
            nbInput = self.layers[-1].size()
        self.layers.append(AveragePool2D(nbInput, pool_size, strides, padding))
    
    def addAveragePool1DLayer(self, pool_size=2, strides=None, padding=False, nbInput=None) -> None:
        if nbInput is None:
            nbInput = self.layers[-1].size()
        self.layers.append(AveragePool1D(nbInput, pool_size, strides, padding))
    

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
        for _ in range(epochs):
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