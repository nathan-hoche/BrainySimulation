from Brain.neuron import neuron

# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

class layers():
    def __init__(self, actFunc, lossFunc, nbNeuron, nbInput) -> None:
        self.output = []
        self.inputList = []
        self.neurons = []
        for _ in range(0, nbNeuron):
            self.neurons.append(neuron(nbInput, actFunc, lossFunc))
        pass

    def size(self) -> int:
        return len(self.neurons)

    def calc(self, inputList: list[float]) -> list[float]:
        self.inputList = inputList
        self.output = []
        for neuron in self.neurons:
            self.output.append(neuron.calc(inputList))
        return self.output
    
    def updateWeight(self, outputExpected: float) -> None:
        if type(outputExpected) is list:
            for i in range(len(self.neurons)):
                self.neurons[i].updateWeight(self.output[i], outputExpected[i], self.inputList)
        else:
            for i in range(len(self.neurons)):
                self.neurons[i].updateWeight(self.output[i], outputExpected, self.inputList)
        
    def printWeight(self) -> None:
        self.neurons[0].printWeight()


class network():
    def __init__(self) -> None:
        self.layers = []
        pass

    def addNewLayer(self, actFunc, lossFunc, nbNeuron, nbInput=None) -> None:
        if nbInput is None:
            nbInput = self.layers[-1].size()
        newLayer = layers(actFunc, lossFunc, nbNeuron, nbInput)
        self.layers.append(newLayer)
    
    def calc(self, inputList: list[float]) -> list[float]:
        output = inputList
        for layer in self.layers:
            output = layer.calc(output)
        return output
    
    def train(self, inputList: list[float], outputExpected: float) -> None:
        self.calc(inputList)
        for layer in reversed(self.layers):
            layer.updateWeight(outputExpected)
    
    def printWeight(self):
        self.layers[0].printWeight()