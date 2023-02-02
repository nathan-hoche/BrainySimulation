from BrainV1.neuron import neuron

class layers():
    def __init__(self, actFunc, lossFunc, nbNeuron, nbInput, learningRate=1.0) -> None:
        self.output = []
        self.inputList = []
        self.neurons = []
        for _ in range(0, nbNeuron):
            self.neurons.append(neuron(nbInput, actFunc, lossFunc, learningRate))
        pass

    def size(self) -> int:
        return len(self.neurons)

    def calc(self, inputList: list[float]) -> list[float]:
        self.inputList = inputList
        self.output = []
        for neuron in self.neurons:
            self.output.append(neuron.calc(inputList))
        return self.output
    
    def updateWeight(self, outputExpected: float|list[float], velocity:list[float]) -> None:
        v = []
        if type(outputExpected) is list:
            for i in range(len(self.neurons)):
                for x in range(len(velocity)):
                    v.append(self.neurons[i].updateWeight(self.output[i], outputExpected[i], self.inputList, velocity[x]))
        else:
            for i in range(len(self.neurons)):
                for x in range(len(velocity)):
                    v.append(self.neurons[i].updateWeight(self.output[i], outputExpected, self.inputList, velocity[x]))
        return v

    def printWeight(self) -> None:
        self.neurons[0].printWeight()

class network():
    def __init__(self, learningRate=1.0) -> None:
        self.layers = []
        self.learningRate = learningRate
        pass

    def addNewLayer(self, actFunc, lossFunc, nbNeuron, nbInput=None) -> None:
        if nbInput is None:
            nbInput = self.layers[-1].size()
        newLayer = layers(actFunc, lossFunc, nbNeuron, nbInput, self.learningRate)
        self.layers.append(newLayer)

    def calc(self, inputList: list[float]) -> list[float]:
        output = inputList
        for layer in self.layers:
            output = layer.calc(output)
        return output

    def train(self, inputList: list[float], outputExpected: float|list[float]) -> None:
        self.calc(inputList)
        velocity = [0]
        if (type(outputExpected) is list):
            velocity = [0] * len(outputExpected)
        for layer in reversed(self.layers):
            velocity = layer.updateWeight(outputExpected, velocity)
    
    def printWeight(self):
        self.layers[0].printWeight()