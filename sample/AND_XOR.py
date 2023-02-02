from BrainV2.utils import Activation
from BrainV2.utils import Gradient
from BrainV2.network import network

class sample():
    def __init__(self) -> None:
        # self.AND = network()
        # self.AND.addNewLayer(Activation.basic, Gradient.basic, 1, 2)
        self.XOR = network(learningRate=0.1)
        self.XOR.addNewLayer(Activation.sigmoid, Gradient.basic, 16, 2)
        self.XOR.addNewLayer(Activation.sigmoid, Gradient.basic, 1)
        self.XOR.printLayers()

    def train(self, isTest:bool=False):
        if isTest == True:
            return
        for _ in range(10000):
            self.XOR.train([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])
        self.XOR.displayLoss()

    def test(self, isTest:bool=False):
        if isTest == True:
            return
        print("Result - AND:")
        # print("\tTest: [1, 1] -> ", self.AND.calc([1, 1]))
        # print("\tTest: [0, 0] -> ", self.AND.calc([0, 0]))
        # print("\tTest: [1, 0] -> ", self.AND.calc([1, 0]))
        # print("\tTest: [0, 1] -> ", self.AND.calc([0, 1]))
        print("Result - XOR:")
        print("\tTest: [1, 1] -> ", self.XOR.calc([1, 1]))
        print("\tTest: [0, 0] -> ", self.XOR.calc([0, 0]))
        print("\tTest: [1, 0] -> ", self.XOR.calc([1, 0]))
        print("\tTest: [0, 1] -> ", self.XOR.calc([0, 1]))

        self.XOR.plot()
