from Brain.neuron import neuron
from Brain.utils import Activation
from Brain.utils import Gradient
from Brain.network import network

class sample():
    def __init__(self) -> None:
        self.network = network()
        self.network.addNewLayer(Activation.basic, Gradient.basic, 2, 4)
        self.network.addNewLayer(Activation.basic, Gradient.basic, 1)

    def train(self, isTest:bool=False):
        if isTest == True:
            return
        print("Training Values:")
        print("\t[1, 1, 1, 1] -> 1")
        print("\t[0, 0, 0, 0] -> 0")
        print("\t[1, 0, 0, 0] -> 1")
        print("\t[0, 1, 0, 0] -> 1")
        print("\t[0, 0, 1, 0] -> 1")
        print("\t[0, 0, 0, 1] -> 1")
        for _ in range(1000):
            self.network.train([1, 1, 1, 1], 1)
            self.network.train([0, 0, 0, 0], 0)
            self.network.train([1, 0, 0, 0], 1)
            self.network.train([0, 1, 0 ,0], 1)
            self.network.train([0, 0, 1, 0], 1)
            self.network.train([0, 0, 0, 1], 1)


    def test(self, isTest:bool=False):
        if isTest == True:
            return
        print("Result - Trained value:")
        print("\tTest: [1, 1, 1, 1] -> ", self.network.calc([1, 1, 1, 1]))
        print("\tTest: [0, 0, 0, 0] -> ", self.network.calc([0, 0, 0, 0]))
        print("\tTest: [1, 0, 0, 0] -> ", self.network.calc([1, 0, 0, 0]))
        print("\tTest: [0, 1, 0, 0] -> ", self.network.calc([0, 1, 0, 0]))
        print("\tTest: [0, 0, 1, 0] -> ", self.network.calc([0, 0, 1, 0]))
        print("\tTest: [0, 0, 0, 1] -> ", self.network.calc([0, 0, 0, 1]))
        print("Result - Untrained value:")
        print("\tTest: [1, 0, 1, 1] -> ", self.network.calc([1, 0, 1, 1]))
        print("\tTest: [1, 0, 0, 1] -> ", self.network.calc([1, 0, 0, 1]))
        print("\tTest: [0, 1, 0, 1] -> ", self.network.calc([0, 1, 0, 1]))
        print("\tTest: [1, 0, 1, 0] -> ", self.network.calc([1, 0, 1, 0]))
        print("\tTest: [1, 1, 1, 0] -> ", self.network.calc([1, 1, 1, 0]))

