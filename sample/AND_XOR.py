from Brain.utils import Activation
from Brain.utils import Gradient
from Brain.network import network

class sample():
    def __init__(self) -> None:
        self.AND = network()
        self.AND.addNewLayer(Activation.basic, Gradient.basic, 1, 2)
        self.XOR = network(learningRate=0.01)
        self.XOR.addNewLayer(Activation.reLu, Gradient.sgd, 16, 2)
        self.XOR.addNewLayer(Activation.basic, Gradient.basic, 1)

    def train(self, isTest:bool=False):
        if isTest == True:
            return
        for _ in range(2000):
            self.AND.train([0, 0], 0)
            self.AND.train([0, 1], 0)
            self.AND.train([1, 0], 0)
            self.AND.train([1, 1], 1)

            self.XOR.train([0, 0], 0)
            self.XOR.train([0, 1], 1)
            self.XOR.train([1, 0], 1)
            self.XOR.train([1, 1], 0)


    def test(self, isTest:bool=False):
        if isTest == True:
            return
        print("Result - AND:")
        print("\tTest: [1, 1] -> ", self.AND.calc([1, 1]))
        print("\tTest: [0, 0] -> ", self.AND.calc([0, 0]))
        print("\tTest: [1, 0] -> ", self.AND.calc([1, 0]))
        print("\tTest: [0, 1] -> ", self.AND.calc([0, 1]))
        print("Result - XOR:")
        print("\tTest: [1, 1] -> ", self.XOR.calc([1, 1]))
        print("\tTest: [0, 0] -> ", self.XOR.calc([0, 0]))
        print("\tTest: [1, 0] -> ", self.XOR.calc([1, 0]))
        print("\tTest: [0, 1] -> ", self.XOR.calc([0, 1]))
